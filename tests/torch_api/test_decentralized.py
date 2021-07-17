import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.internal.common_utils import find_free_port
import unittest
import torch.multiprocessing as mp
import os
from bagua.torch_api.utils import flatten, unflatten
import bagua.torch_api as bagua


N_EPOCHS = 10


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


def run_model(
    rank, nprocs, hierarchical, peer_selection_mode, communication_interval, results
):
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
        bagua.algorithms.decentralized.DecentralizedAlgorithm(
            hierarchical=hierarchical,
            peer_selection_mode=peer_selection_mode,
        ),
        communication_interval=communication_interval,
    )

    ret = results[rank]

    ret.init_weight.copy_(flatten([param.data for param in model.parameters()]))

    for epoch in range(N_EPOCHS):
        data = torch.randn(4, 2).cuda()
        target = torch.randn(4, 4).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

    ret.bucket_weight.copy_(model.bagua_buckets[0]._peer_weight)


def run_torch_model(
    rank,
    nprocs,
    hierarchical,
    peer_selection_mode,
    communication_interval,
    results,
    backend,
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
    model = DecentralizedAlgor(
        model, optimizer, hierarchical, peer_selection_mode, communication_interval
    )

    ret = results[rank]
    ret.init_weight.copy_(flatten([param.data for param in model.parameters()]))

    for epoch in range(N_EPOCHS):
        data = torch.randn(4, 2).cuda()
        target = torch.randn(4, 4).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        model.step()

    ret.bucket_weight.copy_(model.peer_weight)


class Result(object):
    def __init__(self):
        model = Net()
        self.init_weight = flatten(
            [torch.zeros_like(param.data) for param in model.parameters()]
        )
        self.bucket_weight = flatten(
            [torch.zeros_like(param.data) for param in model.parameters()]
        )


class DecentralizedAlgor(nn.Module):
    def __init__(
        self,
        module,
        optimizer,
        hierarchical,
        peer_selection_mode,
        communication_interval,
    ):
        super(DecentralizedAlgor, self).__init__()
        self.module = module
        self.optimizer = optimizer
        self.hierarchical = hierarchical
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval
        self.step_count = 0

        assert torch.distributed.is_initialized()

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        # broadcast parameters
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data, src=0)

    def _build_params(self):
        return [param.data for param in list(self.module.parameters()).__reversed__()]

    def communicate_with_peer(self):
        if self.peer_selection_mode == "all":
            torch.distributed.all_reduce(self.peer_weight)
            self.peer_weight /= self.world_size
        elif self.peer_selection_mode == "shift_one":
            peer_rank = get_peer_rank(
                self.peer_selection_mode,
                self.rank,
                self.world_size,
                self.step_count,
                self.communication_interval,
            )

            weight = self.weight.cpu()
            peer_weight = self.peer_weight.cpu()

            requests = []
            requests.append(torch.distributed.isend(weight, peer_rank))
            requests.append(torch.distributed.irecv(peer_weight, peer_rank))

            for req in requests:
                req.wait()

            self.peer_weight = peer_weight.cuda()
            self.weight = weight.cuda()

            self.peer_weight += self.weight
            self.peer_weight /= 2
        else:
            raise ValueError("Unsupported `peer_selection_mode`")

    def forward(self, *inputs, **kwargs):
        if self.step_count % self.communication_interval == 0:
            self.weight = flatten(self._build_params())
            self.peer_weight = flatten(self._build_params())
            self.communicate_with_peer()

        result = self.module(*inputs, **kwargs)
        return result

    def step(self):
        if self.step_count % self.communication_interval == 0:
            params = self._build_params()
            for buf, synced in zip(params, unflatten(self.peer_weight, params)):
                buf.copy_(synced)

        self.optimizer.step()
        self.step_count += 1


def get_peer_rank(peer_selection_mode, rank, nranks, step, communication_interval):
    comm_step = step // communication_interval
    if peer_selection_mode == "shift_one":
        if rank < nranks // 2:
            return ((comm_step + rank) % ((nranks + 1) // 2)) + (nranks // 2)
        else:
            return (rank - (nranks // 2) - comm_step) % (nranks // 2)
    else:
        ValueError("Unsupported `peer_selection_mode`")


class TestLowPrecisionDecentralized(unittest.TestCase):
    def run_test_locally(
        self, nprocs, hierarchical, peer_selection_mode, communication_interval
    ):
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
            args=(
                nprocs,
                hierarchical,
                peer_selection_mode,
                communication_interval,
                results,
            ),
        )

        for rank in range(nprocs):
            if peer_selection_mode == "all":
                peer_rank = (rank + 1) % nprocs
                # all workers have equal weights
                self.assertTrue(
                    torch.equal(
                        results[rank].bucket_weight,
                        results[peer_rank].bucket_weight,
                    )
                )
            elif peer_selection_mode == "shift_one":
                peer_rank = get_peer_rank(
                    peer_selection_mode,
                    rank,
                    nprocs,
                    N_EPOCHS - 1,
                    communication_interval,
                )

                self.assertTrue(
                    torch.equal(
                        results[rank].bucket_weight, results[peer_rank].bucket_weight
                    )
                )
            else:
                raise ValueError("illegal `peer_selection_mode`!")

    def run_diff_locally(
        self, nprocs, hierarchical, peer_selection_mode, communication_interval, backend
    ):
        if not torch.cuda.is_available():
            print("skip tests since cuda is not available")
            return

        os.environ["WORLD_SIZE"] = str(nprocs)
        os.environ["LOCAL_WORLD_SIZE"] = str(nprocs)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["BAGUA_SERVICE_PORT"] = str(find_free_port())

        torch_results = [Result() for _ in range(nprocs)]
        mp.spawn(
            run_torch_model,
            nprocs=nprocs,
            args=(
                nprocs,
                hierarchical,
                peer_selection_mode,
                communication_interval,
                torch_results,
                backend,
            ),
        )

        bagua_results = [Result() for _ in range(nprocs)]
        mp.spawn(
            run_model,
            nprocs=nprocs,
            args=(
                nprocs,
                hierarchical,
                peer_selection_mode,
                communication_interval,
                bagua_results,
            ),
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
        self.run_test_locally(
            nprocs=8,
            hierarchical=False,
            peer_selection_mode="all",
            communication_interval=1,
        )
        self.run_test_locally(
            nprocs=8,
            hierarchical=False,
            peer_selection_mode="shift_one",
            communication_interval=1,
        )
        self.run_test_locally(
            nprocs=8,
            hierarchical=False,
            peer_selection_mode="shift_one",
            communication_interval=2,
        )

    def test_compare(self):
        self.run_diff_locally(
            nprocs=8,
            hierarchical=False,
            peer_selection_mode="all",
            communication_interval=1,
            backend="gloo",
        )
        self.run_diff_locally(
            nprocs=8,
            hierarchical=False,
            peer_selection_mode="shift_one",
            communication_interval=1,
            backend="gloo",
        )
        self.run_diff_locally(
            nprocs=8,
            hierarchical=False,
            peer_selection_mode="shift_one",
            communication_interval=2,
            backend="gloo",
        )


if __name__ == "__main__":
    unittest.main()