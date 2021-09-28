import bagua.torch_api as bagua
import torch
import unittest
from tests.internal.multi_process import MultiProcessTestCase, setup_bagua_env
from tests import skip_if_cuda_not_available


class Result(object):
    def __init__(self):
        self.data = torch.zeros(100)


def run_new_group(rank, nprocs, args, results, env):
    setup_bagua_env(rank, env)

    all_ranks = list(range(nprocs))
    odd_ranks = list(filter(lambda r: r % 2 == 1, all_ranks))
    g = bagua.communication.new_group(ranks=odd_ranks)

    tensor = torch.rand(100).cuda()
    tensor *= rank

    bagua.communication.allreduce(tensor, tensor, comm=g.get_global_communicator())
    results[rank].data.copy_(tensor)


def run_from_torch_group(rank, nprocs, args, results, env):
    setup_bagua_env(rank, env)

    all_ranks = list(range(nprocs))
    ranks_1 = list(filter(lambda r: r % 3 == 1, all_ranks))
    ranks_2 = list(filter(lambda r: r % 2 == 0, all_ranks))

    g_1 = torch.distributed.new_group(ranks_1)
    bg_1 = bagua.communication.from_torch_group(g_1)

    g_2 = torch.distributed.new_group(ranks_2)
    bg_2 = bagua.communication.from_torch_group(g_2)

    if rank in ranks_1:
        assert torch.distributed.get_rank(g_1) == bg_1.get_global_communicator().rank()
        assert (
            torch.distributed.get_world_size(g_1)
            == bg_1.get_global_communicator().nranks()  # noqa: W503
        )

    if rank in ranks_2:
        assert torch.distributed.get_rank(g_2) == bg_2.get_global_communicator().rank()
        assert (
            torch.distributed.get_world_size(g_2)
            == bg_2.get_global_communicator().nranks()  # noqa: W503
        )


class TestProcessGroup(MultiProcessTestCase):
    @skip_if_cuda_not_available()
    def test_new_group(self):
        nprocs = torch.cuda.device_count()
        results = [Result() for _ in range(nprocs)]
        self.run_test_locally(run_new_group, nprocs, args={}, results=results)

        all_ranks = list(range(nprocs))
        odd_ranks = list(filter(lambda r: r % 2 == 1, all_ranks))

        for rank in odd_ranks:
            peer_rank = (rank + 2) % nprocs

            self.assertTrue(torch.equal(results[rank].data, results[peer_rank].data))

    @skip_if_cuda_not_available()
    def test_from_torch_group(self):
        nprocs = torch.cuda.device_count()
        self.run_test_locally(run_from_torch_group, nprocs, args={}, results=None)


if __name__ == "__main__":
    unittest.main()
