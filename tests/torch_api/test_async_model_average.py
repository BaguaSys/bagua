import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.internal.common_utils import find_free_port
import unittest
import torch.multiprocessing as mp
import os
import bagua.torch_api as bagua


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


def run_model(rank):
    # initialize subprocess env
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # init bagua distributed process group
    torch.cuda.set_device(rank)
    bagua.init_process_group()

    # construct model and optimizer, etc.
    model = Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # wrap model
    algorithm = bagua.algorithms.async_model_average.AsyncModelAverageAlgorithm()
    model = model.with_bagua(
        [optimizer], algorithm
    )

    for _ in range(10):
        data = torch.randn(4, 2).cuda()
        target = torch.randn(4, 4).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

    algorithm.abort(model)


class TestAsyncModelAverage(unittest.TestCase):
    def test_algorithm(self):
        if not torch.cuda.is_available():
            print("skip tests since cuda is not available")
            return

        nprocs = torch.cuda.device_count()
        os.environ["WORLD_SIZE"] = str(nprocs)
        os.environ["LOCAL_WORLD_SIZE"] = str(nprocs)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["BAGUA_SERVICE_PORT"] = str(find_free_port())

        mp.spawn(
            run_model,
            nprocs=nprocs,
        )


if __name__ == "__main__":
    unittest.main()
