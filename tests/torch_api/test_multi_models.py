import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.internal.common_utils import find_free_port
import unittest
import multiprocessing
import os
from bagua.torch_api.utils import flatten
import bagua.torch_api as bagua
from tests import skip_if_cuda_not_available

N_EPOCHS = 10


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=True)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 30, bias=True)
        self.fc3 = nn.Linear(30, 20, bias=True)
        self.fc4 = nn.Linear(20, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)


def _init_bagua_env(rank, env):
    # set deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(rank)
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


def _init_torch_env(rank, nprocs, backend):
    # set deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(rank)

    # init torch distributed process group
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        world_size=nprocs,
        rank=rank,
        backend=backend,
        init_method="file:///tmp/.bagua.test.filestore",
    )


def run_model(
    rank,
    results,
    env,
):
    _init_bagua_env(rank, env)

    # construct model and optimizer, etc.
    model_1 = Net1().cuda()
    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.01)
    loss_fn_1 = nn.MSELoss()

    model_2 = Net2().cuda()
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.01)
    loss_fn_2 = nn.MSELoss()

    # wrap model
    from bagua.torch_api.algorithms import gradient_allreduce

    algorithm = gradient_allreduce.GradientAllReduceAlgorithm()

    model_1 = model_1.with_bagua([optimizer_1], algorithm)

    model_2 = model_2.with_bagua([optimizer_2], algorithm)

    ret = results[rank]

    ret.init_weight_1.copy_(flatten([param.data for param in model_1.parameters()]))
    ret.init_weight_2.copy_(flatten([param.data for param in model_2.parameters()]))

    for epoch in range(N_EPOCHS):
        data_1 = torch.randn(8, 2).cuda()
        target_1 = torch.randn(8, 4).cuda()

        optimizer_1.zero_grad()
        output_1 = model_1(data_1)
        loss_1 = loss_fn_1(output_1, target_1)

        loss_1.backward()
        optimizer_1.step()

        data_2 = torch.randn(8, 2).cuda()
        target_2 = torch.randn(8, 4).cuda()

        optimizer_2.zero_grad()
        output_2 = model_2(data_2)
        loss_2 = loss_fn_2(output_2, target_2)

        loss_2.backward()
        optimizer_2.step()

    ret.end_weight_1.copy_(flatten([param.data for param in model_1.parameters()]))
    ret.end_weight_2.copy_(flatten([param.data for param in model_2.parameters()]))


def run_torch_model(
    rank,
    nprocs,
    results,
    backend,
    env,
):
    _init_torch_env(rank, nprocs, backend)

    # construct model and optimizer, etc.
    model_1 = Net1().cuda()
    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.01)
    loss_fn_1 = nn.MSELoss()

    model_2 = Net2().cuda()
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.01)
    loss_fn_2 = nn.MSELoss()

    # wrap model
    model_1 = torch.nn.parallel.DistributedDataParallel(model_1, device_ids=[rank])
    model_2 = torch.nn.parallel.DistributedDataParallel(model_2, device_ids=[rank])

    ret = results[rank]

    ret.init_weight_1.copy_(flatten([param.data for param in model_1.parameters()]))
    ret.init_weight_2.copy_(flatten([param.data for param in model_2.parameters()]))

    for epoch in range(N_EPOCHS):
        data_1 = torch.randn(8, 2).cuda()
        target_1 = torch.randn(8, 4).cuda()

        optimizer_1.zero_grad()
        output_1 = model_1(data_1)
        loss_1 = loss_fn_1(output_1, target_1)

        loss_1.backward()
        optimizer_1.step()

        data_2 = torch.randn(8, 2).cuda()
        target_2 = torch.randn(8, 4).cuda()

        optimizer_2.zero_grad()
        output_2 = model_2(data_2)
        loss_2 = loss_fn_2(output_2, target_2)

        loss_2.backward()
        optimizer_2.step()

    ret.end_weight_1.copy_(flatten([param.data for param in model_1.parameters()]))
    ret.end_weight_2.copy_(flatten([param.data for param in model_2.parameters()]))


class Result(object):
    def __init__(self):
        model_1 = Net1()
        model_2 = Net2()
        self.init_weight_1 = flatten(
            [torch.zeros_like(param.data) for param in model_1.parameters()]
        )
        self.end_weight_1 = flatten(
            [torch.zeros_like(param.data) for param in model_1.parameters()]
        )

        self.init_weight_2 = flatten(
            [torch.zeros_like(param.data) for param in model_2.parameters()]
        )
        self.end_weight_2 = flatten(
            [torch.zeros_like(param.data) for param in model_2.parameters()]
        )


class TestMultiModels(unittest.TestCase):
    @skip_if_cuda_not_available()
    def test_multi_models(self):
        nprocs = torch.cuda.device_count()
        env = {}
        mp = multiprocessing.get_context("spawn")
        torch_results = [Result() for _ in range(nprocs)]
        processes = []
        backend = "gloo"
        for i in range(nprocs):
            p = mp.Process(
                target=run_torch_model,
                args=(
                    i,
                    nprocs,
                    torch_results,
                    backend,
                    env,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=60)
            self.assertTrue(p.exitcode == 0)

        env = {
            "WORLD_SIZE": str(nprocs),
            "LOCAL_WORLD_SIZE": str(nprocs),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(find_free_port(8000, 8100)),
            "BAGUA_SERVICE_PORT": str(find_free_port(9000, 9100)),
        }

        bagua_results = [Result() for _ in range(nprocs)]
        processes = []
        for i in range(nprocs):
            p = mp.Process(
                target=run_model,
                args=(
                    i,
                    bagua_results,
                    env,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=60)
            self.assertTrue(p.exitcode == 0)

        for rank in range(nprocs):
            self.assertTrue(
                torch.all(
                    torch.isclose(
                        bagua_results[rank].init_weight_1,
                        torch_results[rank].init_weight_1,
                    )
                ).item()
            )

            self.assertTrue(
                torch.all(
                    torch.isclose(
                        bagua_results[rank].end_weight_1,
                        torch_results[rank].end_weight_1,
                    )
                ).item()
            )
            self.assertTrue(
                torch.all(
                    torch.isclose(
                        bagua_results[rank].init_weight_2,
                        torch_results[rank].init_weight_2,
                    )
                ).item()
            )

            self.assertTrue(
                torch.all(
                    torch.isclose(
                        bagua_results[rank].end_weight_2,
                        torch_results[rank].end_weight_2,
                    )
                ).item()
            )


if __name__ == "__main__":
    unittest.main()
