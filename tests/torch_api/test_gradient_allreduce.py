import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.internal.common_utils import find_free_port
from tests.internal.multi_process import setup_bagua_env
import unittest
import multiprocessing
from bagua.torch_api.utils import flatten
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


def _init_bagua_env(rank, env):
    # set deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(rank)
    # initialize subprocess env
    setup_bagua_env(rank, env)


def run_model(
    rank,
    nprocs,
    hierarchical,
    results,
    env,
):
    _init_bagua_env(rank, env)

    # construct model and optimizer, etc.
    model = Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    def run_epochs(num_epochs):
        for epoch in range(num_epochs):
            data = torch.randn(4, 2).cuda()
            target = torch.randn(4, 4).cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)

            loss.backward()
            optimizer.step()

    run_epochs(1)

    # wrap model
    model = model.with_bagua(
        [optimizer],
        bagua.algorithms.gradient_allreduce.GradientAllReduceAlgorithm(
            hierarchical=hierarchical,
        ),
    )

    run_epochs(10)

    ret = results[rank]

    ret._weight.copy_(flatten([param.data for param in model.parameters()]))


class Result(object):
    def __init__(self):
        model = Net()
        self._weight = flatten(
            [torch.zeros_like(param.data) for param in model.parameters()]
        )


class TestGradientAllReduce(unittest.TestCase):
    def run_test_locally(
        self,
        nprocs,
        hierarchical,
    ):
        env = {
            "WORLD_SIZE": str(nprocs),
            "LOCAL_WORLD_SIZE": str(nprocs),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(find_free_port(8000, 8100)),
            "BAGUA_SERVICE_PORT": str(find_free_port(9000, 9100)),
        }

        mp = multiprocessing.get_context("spawn")
        results = [Result() for _ in range(nprocs)]
        processes = []
        for i in range(nprocs):
            p = mp.Process(
                target=run_model,
                args=(
                    i,
                    nprocs,
                    hierarchical,
                    results,
                    env,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=60)
            self.assertTrue(p.exitcode == 0)

        for rank in range(nprocs):
            peer_rank = (rank + 1) % nprocs
            # all workers have equal weights
            self.assertTrue(
                torch.equal(
                    results[rank]._weight,
                    results[peer_rank]._weight,
                )
            )

    @skip_if_cuda_not_available()
    def test_algorithm(self):
        nprocs = torch.cuda.device_count()
        self.run_test_locally(
            nprocs=nprocs,
            hierarchical=False,
        )


if __name__ == "__main__":
    unittest.main()
