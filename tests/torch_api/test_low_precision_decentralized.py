import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.internal.common_utils import find_free_port
import bagua.torch_api as bagua
import unittest
import torch.multiprocessing as mp
import os


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


def run_model(gpu, nprocs, hierarchical, communication_interval, results):
    # initialize subprocess env
    os.environ["RANK"] = str(gpu)
    os.environ["LOCAL_RANK"] = str(gpu)

    # init bagua distributed process group
    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()

    print(
        f"initialize bagua training process, rank: {bagua.get_rank()}, world_size: {bagua.get_world_size()}"
    )

    # construct model and optimizer, etc.
    model = Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # wrap model
    model = model.with_bagua(
        [optimizer],
        bagua.algorithms.decentralized.LowPrecisionDecentralizedAlgorithm(
            hierarchical=hierarchical, communication_interval=communication_interval
        ),
    )

    for batch_idx in range(10):
        data = torch.randn(4, 2).cuda()
        target = torch.randn(4, 4).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

    ret = results[gpu]
    for bucket in model.bagua_buckets:
        ret.bucket_weight = torch.norm(bucket.flattened_tensor())
        ret.weight = torch.norm(bucket._weight)
        ret.left_peer_weight = torch.norm(bucket._left_peer_weight)
        ret.right_peer_weight = torch.norm(bucket._right_peer_weight)


class Result(object):
    def __init__(self):
        self.bucket_weight = 0
        self.weight = 0
        self.left_peer_weight = 0
        self.right_peer_weight = 0


class TestLowPrecisionDecentralized(unittest.TestCase):
    def run_test_locally(self, hierarchical, communication_interval):
        if not torch.cuda.is_available():
            print("skip tests since cuda is not available")
            return

        nprocs = torch.cuda.device_count()
        os.environ["WORLD_SIZE"] = str(nprocs)
        os.environ["LOCAL_WORLD_SIZE"] = str(nprocs)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["BAGUA_SERVICE_PORT"] = str(find_free_port())

        results = [Result() for _ in range(nprocs)]
        mp.spawn(
            run_model,
            nprocs=nprocs,
            args=(nprocs, hierarchical, communication_interval, results),
        )

        for rank in range(nprocs):
            left_peer_rank = (rank + nprocs - 1) % nprocs
            right_peer_rank = (rank + 1) % nprocs

            if hierarchical:
                # all workers have equal weights
                self.assertTrue(
                    results[rank].bucket_weight == results[left_peer_rank].bucket_weight
                )
            else:
                self.assertTrue(
                    results[rank].weight == results[left_peer_rank].right_peer_weight
                )
                self.assertTrue(
                    results[rank].weight == results[right_peer_rank].left_peer_weight
                )

    def test_algorithm(self):
        self.run_test_locally(hierarchical=False, communication_interval=1)
        self.run_test_locally(hierarchical=False, communication_interval=2)
        self.run_test_locally(hierarchical=True, communication_interval=1)


if __name__ == "__main__":
    unittest.main()
