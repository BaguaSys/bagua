import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.internal.common_utils import find_free_port
import unittest
import multiprocessing
import os
from bagua.torch_api.communication import new_group
import bagua.torch_api as bagua
from tests import skip_if_cuda_not_available


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


def run_model_wrapper(rank, env, algorithm):
    # initialize subprocess env
    os.environ["WORLD_SIZE"] = env["WORLD_SIZE"]
    os.environ["LOCAL_WORLD_SIZE"] = env["LOCAL_WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = env["MASTER_ADDR"]
    os.environ["MASTER_PORT"] = env["MASTER_PORT"]
    os.environ["BAGUA_SERVICE_PORT"] = env["BAGUA_SERVICE_PORT"]
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # init bagua distributed process group
    torch.cuda.set_device(rank)
    bagua.init_process_group()
    partial_ranks = [i for i in range(bagua.get_world_size() - 1)]
    partial_group = bagua.communication.new_group(ranks=partial_ranks)

    # construct model and optimizer, etc.
    model = Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # wrap model
    if algorithm == "gradient_allreduce":
        from bagua.torch_api.algorithms import gradient_allreduce

        bagua_algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
    elif algorithm == "bytegrad":
        from bagua.torch_api.algorithms import bytegrad

        bagua_algorithm = bytegrad.ByteGradAlgorithm()
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
        bagua_algorithm = QAdamAlgorithm(optimizer, hierarchical_reduce=False)
    else:
        raise ValueError("unsupported algorithm")

    model = model.with_bagua([optimizer], bagua_algorithm, process_group=partial_group)

    for _ in range(1000):
        data = torch.randn(4, 2).cuda()
        target = torch.randn(4, 4).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

    if algorithm == "async":
        model.bagua_algorithm.abort(model)


class TestBaguaModule(unittest.TestCase):
    def run_algorithm(self, algorithm, nprocs, local_size):
        env = {
            "WORLD_SIZE": str(nprocs),
            "LOCAL_WORLD_SIZE": str(local_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(find_free_port(8000, 8100)),
            "BAGUA_SERVICE_PORT": str(find_free_port(9000, 9100)),
        }

        mp = multiprocessing.get_context("spawn")
        processes = []
        for i in range(nprocs):
            p = mp.Process(target=run_model_wrapper, args=(i, env, algorithm))
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=60)
            self.assertTrue(p.exitcode == 0)

    @skip_if_cuda_not_available()
    def test_gradient_allreduce(self):
        nprocs = torch.cuda.device_count()
        self.run_algorithm("gradient_allreduce", nprocs, nprocs)

    @skip_if_cuda_not_available()
    def test_bytegrad(self):
        return
        nprocs = torch.cuda.device_count()
        self.run_algorithm("bytegrad", nprocs, 1)

    @skip_if_cuda_not_available()
    def test_decentralized(self):
        nprocs = torch.cuda.device_count()
        self.run_algorithm("decentralized", nprocs, nprocs)

    @skip_if_cuda_not_available()
    def test_low_prec_decentralized(self):
        nprocs = torch.cuda.device_count()
        self.run_algorithm("low_prec_decentralized", nprocs, nprocs)

    @skip_if_cuda_not_available()
    def test_async(self):
        nprocs = torch.cuda.device_count()
        self.run_algorithm("async", nprocs, nprocs)

    @skip_if_cuda_not_available()
    def test_qadam(self):
        return
        nprocs = torch.cuda.device_count()
        self.run_algorithm("qadam", nprocs, nprocs)


if __name__ == "__main__":
    unittest.main()
