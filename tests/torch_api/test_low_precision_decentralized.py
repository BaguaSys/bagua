import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.internal.common_utils import find_free_port
from tests.internal.compressor import MinMaxUInt8
import bagua.torch_api as bagua
import unittest
import torch.multiprocessing as mp
import os
from bagua.torch_api.utils import apply_flattened_call, flatten


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


def _init_env(gpu):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(47)
    # initialize subprocess env
    os.environ["RANK"] = str(gpu)
    os.environ["LOCAL_RANK"] = str(gpu)


def run_model(gpu, nprocs, hierarchical, communication_interval, results):
    _init_env(gpu)

    # construct model and optimizer, etc.
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    ret = results[gpu]
    ret.init_weight.copy_(
        torch.norm(flatten([param.data for param in model.parameters()]))
    )

    # init bagua distributed process group
    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()

    print(
        f"initialize bagua training process, rank: {bagua.get_rank()}, world_size: {bagua.get_world_size()}"
    )

    model.cuda()

    # wrap model
    model = model.with_bagua(
        [optimizer],
        bagua.algorithms.decentralized.LowPrecisionDecentralizedAlgorithm(
            hierarchical=hierarchical, communication_interval=communication_interval
        ),
    )

    for _ in range(10):
        data = torch.randn(4, 2).cuda()
        target = torch.randn(4, 4).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

    ret = results[gpu]
    bucket = model.bagua_buckets[0]
    ret.bucket_weight.copy_(flatten([param.data for param in model.parameters()]))
    ret.weight.copy_(torch.norm(bucket._weight))
    ret.left_peer_weight.copy_(torch.norm(bucket._left_peer_weight))
    ret.right_peer_weight.copy_(torch.norm(bucket._right_peer_weight))


def run_torch_model(gpu, nprocs, hierarchical, communication_interval, results):
    _init_env(gpu)

    # construct model and optimizer, etc.
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    ret = results[gpu]
    ret.init_weight.copy_(
        torch.norm(flatten([param.data for param in model.parameters()]))
    )

    # init torch distributed process group
    store = torch.distributed.FileStore("/tmp/filestore", nprocs)
    torch.distributed.init_process_group(
        world_size=nprocs, rank=gpu, store=store, backend="gloo"
    )

    print(
        f"initialize torch training process, rank: {bagua.get_rank()}, world_size: {bagua.get_world_size()}"
    )

    # wrap model
    model = LowPrecDecentralizedAlgor(
        model, optimizer, hierarchical, communication_interval
    )

    for _ in range(10):
        data = torch.randn(4, 2)
        target = torch.randn(4, 4)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        model.step()

    ret.bucket_weight.copy_(flatten([param.data for param in model.parameters()]))


class Result(object):
    def __init__(self):
        model = Net()
        self.init_weight = torch.Tensor([0.0]).share_memory_()
        self.bucket_weight = flatten(
            [param.data for param in model.parameters()]
        ).share_memory_()
        self.weight = torch.Tensor([0.0]).share_memory_()
        self.left_peer_weight = torch.Tensor([0.0]).share_memory_()
        self.right_peer_weight = torch.Tensor([0.0]).share_memory_()


class LowPrecDecentralizedAlgor(nn.Module):
    def __init__(self, module, optimizer, hierarchical, communication_interval):
        super(LowPrecDecentralizedAlgor, self).__init__()
        self.module = module
        self.optimizer = optimizer
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.step_count = 0
        self.compressor = MinMaxUInt8()

        assert torch.distributed.is_initialized()

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        weights = [param.data for param in self.module.parameters()]
        apply_flattened_call(weights, lambda x: torch.distributed.broadcast(x, 0))

        self.weight = flatten(weights)
        self.left_peer_weight = self.weight.detach().clone()
        self.right_peer_weight = self.weight.detach().clone()

    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
        return result

    def step(self):
        self.optimizer.step()

        def allreduce_fn(x):
            torch.distributed.allreduce(x)
            x /= self.world_size

        def communicate_with_peers(_buffer):
            left_buffer = torch.zeros_like(_buffer, device=_buffer.device)
            right_buffer = torch.zeros_like(_buffer, device=_buffer.device)

            left_peer_rank = (self.rank + self.world_size - 1) % self.world_size
            right_peer_rank = (self.rank + 1) % self.world_size

            requests = []
            requests.append(torch.distributed.isend(_buffer, left_peer_rank))
            requests.append(torch.distributed.isend(_buffer, right_peer_rank))
            requests.append(torch.distributed.irecv(left_buffer, left_peer_rank))
            requests.append(torch.distributed.irecv(right_buffer, right_peer_rank))

            for req in requests:
                req.wait()

            return left_buffer, right_buffer

        def update_weight_fn(x):
            diff = (
                x
                + 1 / 3 * self.left_peer_weight
                + 1 / 3 * self.right_peer_weight
                - 5 / 3 * self.weight
            )

            _min, _max, compressed_buffer = self.compressor.compress(diff)

            left_compressed_buffer, right_compressed_buffer = communicate_with_peers(
                compressed_buffer
            )
            left_min, right_min = communicate_with_peers(_min)
            left_max, right_max = communicate_with_peers(_max)

            left_decompressed = self.compressor.decompress(
                left_min, left_max, left_compressed_buffer
            )
            right_decompressed = self.compressor.decompress(
                right_min, right_max, right_compressed_buffer
            )

            self.left_peer_weight += left_decompressed
            self.right_peer_weight += right_decompressed

            decompressed = self.compressor.decompress(_min, _max, compressed_buffer)
            x += decompressed

        if self.step_count % self.communication_interval == 0:
            weights = [param.data for param in self.module.parameters()]
            if self.hierarchical:
                apply_flattened_call(weights, allreduce_fn)
                apply_flattened_call(
                    weights, lambda x: torch.distributed.broadcast(x, 0)
                )
            else:
                apply_flattened_call(weights, update_weight_fn)
                self.weight = flatten(weights)

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
                    results[rank].weight.item()
                    == results[left_peer_rank].right_peer_weight.item()
                )
                self.assertTrue(
                    results[rank].weight.item()
                    == results[right_peer_rank].left_peer_weight.item()
                )

    def run_diff_locally(self, hierarchical, communication_interval):
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
            args=(nprocs, hierarchical, communication_interval, torch_results),
        )

        bagua_results = [Result() for _ in range(nprocs)]
        mp.spawn(
            run_model,
            nprocs=nprocs,
            args=(nprocs, hierarchical, communication_interval, bagua_results),
        )

        for rank in range(nprocs):
            self.assertTrue(
                bagua_results[rank].init_weight.item()
                == torch_results[rank].init_weight.item()
            )

            ret = torch.all(
                torch.isclose(
                    bagua_results[rank].bucket_weight,
                    torch_results[rank].bucket_weight,
                )
            ).item()

            self.assertTrue(ret)

    def test_algorithm(self):
        self.run_test_locally(hierarchical=False, communication_interval=1)
        self.run_test_locally(hierarchical=False, communication_interval=2)
        self.run_test_locally(hierarchical=True, communication_interval=1)

    def test_compare(self):
        self.run_diff_locally(hierarchical=False, communication_interval=1)


if __name__ == "__main__":
    unittest.main()
