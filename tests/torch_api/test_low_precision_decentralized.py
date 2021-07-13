import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.internal.common_utils import find_free_port
from tests.internal.compressor import MinMaxUInt8
import unittest
import torch.multiprocessing as mp
import os
from bagua.torch_api.utils import apply_flattened_call, flatten
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


def _init_env(rank):
    # set deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(rank)
    # initialize subprocess env
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)


def run_model(rank, nprocs, hierarchical, communication_interval, results):
    _init_env(rank)

    # init bagua distributed process group
    torch.cuda.set_device(rank)
    bagua.init_process_group()

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

    ret = results[rank]
    bucket = model.bagua_buckets[0]

    ret.init_weight.copy_(flatten([param.data for param in model.parameters()]))

    for epoch in range(10):
        data = torch.randn(4, 2).cuda()
        target = torch.randn(4, 4).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

    ret.bucket_weight.copy_(flatten([param.data for param in model.parameters()]))
    ret.weight.copy_(torch.norm(bucket._weight))
    ret.left_peer_weight.copy_(torch.norm(bucket._left_peer_weight))
    ret.right_peer_weight.copy_(torch.norm(bucket._right_peer_weight))


def run_torch_model(
    rank, nprocs, hierarchical, communication_interval, results, backend
):
    _init_env(rank)

    # init torch distributed process group
    torch.cuda.set_device(rank)
    store = torch.distributed.FileStore("/tmp/filestore", nprocs)
    torch.distributed.init_process_group(
        world_size=nprocs, rank=rank, store=store, backend=backend
    )

    # construct model and optimizer, etc.
    model = Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # wrap model
    model = LowPrecDecentralizedAlgor(
        model, optimizer, hierarchical, communication_interval
    )

    ret = results[rank]
    ret.init_weight.copy_(flatten([param.data for param in model.parameters()]))

    for epoch in range(10):
        data = torch.randn(4, 2).cuda()
        target = torch.randn(4, 4).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        model.step()

    ret.bucket_weight.copy_(flatten([param.data for param in model.parameters()]))


class Result(object):
    def __init__(self):
        model = Net()
        self.init_weight = flatten(
            [torch.zeros_like(param.data) for param in model.parameters()]
        )
        self.bucket_weight = flatten(
            [torch.zeros_like(param.data) for param in model.parameters()]
        )
        self.weight = torch.Tensor([0.0])
        self.left_peer_weight = torch.Tensor([0.0])
        self.right_peer_weight = torch.Tensor([0.0])


class LowPrecDecentralizedAlgor(nn.Module):
    def __init__(self, module, optimizer, hierarchical, communication_interval):
        super(LowPrecDecentralizedAlgor, self).__init__()
        self.module = module
        self.optimizer = optimizer
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.compressor = MinMaxUInt8()
        self.step_count = 0

        assert torch.distributed.is_initialized()

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        # broadcast parameters
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data, src=0)

        self.weight = flatten(self._build_params()).cuda()
        self.left_peer_weight = self.weight.detach().clone().cuda()
        self.right_peer_weight = self.weight.detach().clone().cuda()

    def _build_params(self):
        return [param.data for param in list(self.module.parameters()).__reversed__()]

    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
        return result

    def step(self):
        self.optimizer.step()

        def communicate_with_peers(
            tensor: torch.Tensor, comm_size: int
        ) -> (torch.Tensor, torch.Tensor):
            if comm_size == 1:
                return tensor, tensor

            tensor = tensor.cpu()
            left_tensor = torch.zeros_like(tensor)
            right_tensor = torch.zeros_like(tensor)

            left_peer_rank = (self.rank + self.world_size - 1) % comm_size
            right_peer_rank = (self.rank + 1) % comm_size

            requests = []
            requests.append(torch.distributed.isend(tensor, left_peer_rank))
            requests.append(torch.distributed.isend(tensor, right_peer_rank))
            requests.append(torch.distributed.irecv(left_tensor, left_peer_rank))
            requests.append(torch.distributed.irecv(right_tensor, right_peer_rank))

            for req in requests:
                req.wait()

            return left_tensor.cuda(), right_tensor.cuda()

        def update_weight_fn(x, comm_size):
            x += 1 / 3 * self.left_peer_weight
            x += 1 / 3 * self.right_peer_weight
            x -= 5 / 3 * self.weight

            minmax, compressed = self.compressor.compress(x)
            left_compressed, right_compressed = communicate_with_peers(
                compressed, comm_size
            )
            left_minmax, right_minmax = communicate_with_peers(minmax, comm_size)

            self.left_peer_weight += self.compressor.decompress(
                left_minmax, left_compressed
            )
            self.right_peer_weight += self.compressor.decompress(
                right_minmax, right_compressed
            )

            diff = self.compressor.decompress(minmax, compressed)
            x.copy_(self.weight + diff)
            self.weight.copy_(x)

        def hierarchical_update_weight_fn(x):
            torch.distributed.reduce(x, dst=0)
            if self.rank == 0:
                x /= self.world_size
                update_weight_fn(x, comm_size=1)

            torch.distributed.broadcast(x, 0)

        if self.step_count % self.communication_interval == 0:
            weights = self._build_params()
            if self.hierarchical:
                apply_flattened_call(weights, hierarchical_update_weight_fn)
            else:
                apply_flattened_call(
                    weights, lambda x: update_weight_fn(x, self.world_size)
                )

        self.step_count += 1


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
                    torch.equal(
                        results[rank].bucket_weight,
                        results[left_peer_rank].bucket_weight,
                    )
                )
            else:
                self.assertTrue(
                    torch.equal(
                        results[rank].weight, results[left_peer_rank].right_peer_weight
                    )
                )
                self.assertTrue(
                    torch.equal(
                        results[rank].weight, results[right_peer_rank].left_peer_weight
                    )
                )

    def run_diff_locally(self, hierarchical, communication_interval, backend):
        if not torch.cuda.is_available():
            print("skip tests since cuda is not available")
            return

        nprocs = torch.cuda.device_count()
        os.environ["WORLD_SIZE"] = str(nprocs)
        os.environ["LOCAL_WORLD_SIZE"] = str(nprocs)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["BAGUA_SERVICE_PORT"] = str(find_free_port())

        torch_results = [Result() for _ in range(nprocs)]
        mp.spawn(
            run_torch_model,
            nprocs=nprocs,
            args=(nprocs, hierarchical, communication_interval, torch_results, backend),
        )

        bagua_results = [Result() for _ in range(nprocs)]
        mp.spawn(
            run_model,
            nprocs=nprocs,
            args=(nprocs, hierarchical, communication_interval, bagua_results),
        )

        for rank in range(nprocs):
            self.assertTrue(
                torch.all(
                    torch.isclose(
                        bagua_results[rank].init_weight,
                        torch_results[rank].init_weight,
                    )
                ).item()
            )

            self.assertTrue(
                torch.all(
                    torch.isclose(
                        bagua_results[rank].bucket_weight,
                        torch_results[rank].bucket_weight,
                    )
                ).item()
            )

    def test_algorithm(self):
        self.run_test_locally(hierarchical=False, communication_interval=1)
        self.run_test_locally(hierarchical=False, communication_interval=2)
        self.run_test_locally(hierarchical=True, communication_interval=1)

    def test_compare(self):
        self.run_diff_locally(
            hierarchical=False, communication_interval=1, backend="gloo"
        )
        self.run_diff_locally(
            hierarchical=False, communication_interval=2, backend="gloo"
        )
        self.run_diff_locally(
            hierarchical=True, communication_interval=1, backend="nccl"
        )


if __name__ == "__main__":
    unittest.main()
