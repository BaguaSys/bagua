import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.internal.common_utils import find_free_port
from tests.internal.multi_process import setup_bagua_env
import unittest
import multiprocessing
from tests import skip_if_cuda_not_available
from bagua.torch_api.data_parallel import DistributedDataParallel as DDP


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=True)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def train(model, optimizer, loss_fn, is_async):
    if is_async:
        model.bagua_algorithm.resume(model)

    for _ in range(1000):
        data = torch.randn(4, 2).cuda()
        target = torch.randn(4, 4).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

    if is_async:
        model.bagua_algorithm.abort(model)


def bagua_init(model, optimizer, algorithm, process_group=None):
    # wrap model
    if algorithm == "gradient_allreduce":
        from bagua.torch_api.algorithms import gradient_allreduce

        bagua_algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
    elif algorithm == "bytegrad":
        from bagua.torch_api.algorithms import bytegrad

        bagua_algorithm = bytegrad.ByteGradAlgorithm(hierarchical=False)
    elif algorithm == "decentralized":
        from bagua.torch_api.algorithms import decentralized

        bagua_algorithm = decentralized.DecentralizedAlgorithm(hierarchical=False)
    elif algorithm == "async":
        from bagua.torch_api.algorithms import async_model_average

        bagua_algorithm = async_model_average.AsyncModelAverageAlgorithm(
            sync_interval_ms=20,
        )
    elif algorithm == "low_prec_decentralized":
        from bagua.torch_api.algorithms import decentralized

        bagua_algorithm = decentralized.LowPrecisionDecentralizedAlgorithm(
            hierarchical=False
        )
    elif algorithm == "qadam":
        from bagua.torch_api.algorithms.q_adam import QAdamAlgorithm, QAdamOptimizer

        optimizer = QAdamOptimizer(model.parameters(), warmup_steps=10)
        bagua_algorithm = QAdamAlgorithm(optimizer, hierarchical=False)
    else:
        raise ValueError("unsupported algorithm")

    ddp_model = DDP(model, optimizers=[optimizer], algorithm=bagua_algorithm, process_group=process_group)

    return ddp_model, optimizer


def run_model_wrapper(rank, nranks, env, algorithm):
    # initialize subprocess env
    setup_bagua_env(rank, env)

    # construct model and optimizer, etc.
    model = Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    ddp_model, optimizer = bagua_init(
        model, optimizer, algorithm
    )

    train(ddp_model, optimizer, loss_fn, is_async=(algorithm == "async"))


def run_model_switch_wrapper(rank, nranks, env, algorithms):
    # initialize subprocess env
    setup_bagua_env(rank, env)

    # construct model and optimizer, etc.
    model = Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for i in range(len(algorithms)):
        ddp_model, optimizer = bagua_init(model, optimizer, algorithms[i])
        train(ddp_model, optimizer, loss_fn, is_async=(algorithms[i] == "async"))
        print('algorithm={} done'.format(algorithms[i]))


class TestBaguaModule(unittest.TestCase):
    def run_algorithm(self, nprocs, nranks, fn, algorithm):
        env = {
            "WORLD_SIZE": str(nprocs),
            "LOCAL_WORLD_SIZE": str(nprocs),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(find_free_port(8000, 8100)),
            "BAGUA_SERVICE_PORT": str(find_free_port(9000, 9100)),
        }

        mp = multiprocessing.get_context("spawn")
        processes = []
        for i in range(nprocs):
            p = mp.Process(target=fn, args=(i, nranks, env, algorithm))
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=60)
            self.assertEqual(p.exitcode, 0)

    @skip_if_cuda_not_available()
    def test_gradient_allreduce(self):
        nprocs = torch.cuda.device_count()
        self.run_algorithm(
            nprocs, nprocs - 1, run_model_wrapper, algorithm="gradient_allreduce"
        )

    @skip_if_cuda_not_available()
    def test_decentralized(self):
        nprocs = torch.cuda.device_count()
        self.run_algorithm(
            nprocs, nprocs - 1, run_model_wrapper, algorithm="decentralized"
        )

    @skip_if_cuda_not_available()
    def test_low_prec_decentralized(self):
        nprocs = torch.cuda.device_count()
        self.run_algorithm(
            nprocs, nprocs - 1, run_model_wrapper, algorithm="low_prec_decentralized"
        )

    @skip_if_cuda_not_available()
    def test_async(self):
        nprocs = torch.cuda.device_count()
        self.run_algorithm(nprocs, nprocs - 1, run_model_wrapper, algorithm="async")

    @skip_if_cuda_not_available()
    def test_bytegrad(self):
        nprocs = torch.cuda.device_count()
        self.run_algorithm(nprocs, nprocs - 1, run_model_wrapper, algorithm="bytegrad")

    @skip_if_cuda_not_available()
    def test_qadam(self):
        nprocs = torch.cuda.device_count()
        self.run_algorithm(nprocs, nprocs - 1, run_model_wrapper, algorithm="qadam")

    @skip_if_cuda_not_available()
    def test_model_switch(self):
        nprocs = torch.cuda.device_count()
        algorithms = [
            "bytegrad",
            "decentralized",
            "gradient_allreduce",
            "qadam",
            "async",
            "low_prec_decentralized",
        ]
        self.run_algorithm(nprocs, nprocs - 1, run_model_switch_wrapper, algorithms)


if __name__ == "__main__":
    unittest.main()
