"""
This script is modified from the horovod benchmark script(https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_synthetic_benchmark.py).
You can compare the performance with the horovod script directly.
"""

import argparse
import timeit
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import numpy as np
import bagua.torch_api as bagua
import logging


logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.ERROR)
if bagua.get_rank() == 0:
    logging.getLogger().setLevel(logging.INFO)

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch Synthetic Benchmark",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--model", type=str, default="resnet50", help="model to benchmark")
parser.add_argument("--batch-size", type=int, default=32, help="input batch size")

parser.add_argument(
    "--num-warmup-batches",
    type=int,
    default=10,
    help="number of warm-up batches that don't count towards benchmark",
)
parser.add_argument(
    "--num-batches-per-iter",
    type=int,
    default=10,
    help="number of batches per benchmark iteration",
)
parser.add_argument(
    "--num-iters", type=int, default=10, help="number of benchmark iterations"
)

parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)

parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="deterministic reproducible training",
)

# bagua args
parser.add_argument(
    "--algorithm",
    type=str,
    default="gradient_allreduce",
    help="gradient_allreduce, bytegrad, decentralized, low_precision_decentralized, qadam or async",
)
parser.add_argument(
    "--async-sync-interval",
    default=500,
    type=int,
    help="Model synchronization interval(ms) for async algorithm",
)
parser.add_argument(
    "--async-warmup-steps",
    default=0,
    type=int,
    help="Warmup(allreduce) steps for async algorithm",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="use amp",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--fuse-optimizer",
    action="store_true",
    default=False,
    help="fuse optimizer or not",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not args.cuda:
    raise RuntimeError("bagua currently not supporting non-GPU mode")

if args.cuda:
    torch.cuda.set_device(bagua.get_local_rank())
bagua.init_process_group()

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

cudnn.benchmark = True
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(bagua.get_rank())
    torch.manual_seed(bagua.get_rank())
    torch.cuda.manual_seed(bagua.get_rank())


# Set up standard model.
model = getattr(models, args.model)()

if args.cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01 * bagua.get_world_size())
if args.fuse_optimizer:
    optimizer = bagua.contrib.fuse_optimizer(optimizer)

if args.algorithm == "gradient_allreduce":
    from bagua.torch_api.algorithms import gradient_allreduce

    algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
elif args.algorithm == "decentralized":
    from bagua.torch_api.algorithms import decentralized

    algorithm = decentralized.DecentralizedAlgorithm()
elif args.algorithm == "low_precision_decentralized":
    from bagua.torch_api.algorithms import decentralized

    algorithm = decentralized.LowPrecisionDecentralizedAlgorithm()
elif args.algorithm == "bytegrad":
    from bagua.torch_api.algorithms import bytegrad

    algorithm = bytegrad.ByteGradAlgorithm()
elif args.algorithm == "qadam":
    from bagua.torch_api.algorithms import q_adam

    optimizer = q_adam.QAdamOptimizer(
        model.parameters(), lr=0.01 * bagua.get_world_size(), warmup_steps=100
    )
    algorithm = q_adam.QAdamAlgorithm(optimizer)
elif args.algorithm == "async":
    from bagua.torch_api.algorithms import async_model_average

    algorithm = async_model_average.AsyncModelAverageAlgorithm(
        sync_interval_ms=args.async_sync_interval,
        warmup_steps=args.async_warmup_steps,
    )
else:
    raise NotImplementedError

model = model.with_bagua([optimizer], algorithm, do_flatten=not args.fuse_optimizer)

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()


batch_idx = 0


def benchmark_step():
    global batch_idx
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)

    if args.amp:
        scaled_loss = args.scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()
        return

    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
        logging.info(
            "BatchIdx: {} TrainLoss: {:.6f}".format(
                batch_idx,
                loss.item(),
            )
        )
    batch_idx += 1


logging.info("Model: %s" % args.model)
logging.info("Batch size: %d" % args.batch_size)
device = "GPU" if args.cuda else "CPU"
logging.info("Number of %ss: %d" % (device, bagua.get_world_size()))

# Warm-up
logging.info("Running warmup...")

timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
logging.info("Running benchmark...")
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    logging.info(
        "Iter #%d: %.1f img/sec %s" % (x, img_sec * bagua.get_world_size(), device)
    )
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
logging.info("Img/sec per %s: %.1f +-%.1f" % (device, img_sec_mean, img_sec_conf))
logging.info(
    "Total img/sec on %d %s(s): %.1f +-%.1f"
    % (
        bagua.get_world_size(),
        device,
        bagua.get_world_size() * img_sec_mean,
        bagua.get_world_size() * img_sec_conf,
    )
)

if args.algorithm == "async":
    model.bagua_algorithm.abort(model)
