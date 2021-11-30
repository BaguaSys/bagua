import os
import unittest

import torch
import torch.distributed as c10d

import bagua.torch_api as bagua
from tests.internal.common_utils import find_free_port
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


from torch.testing._internal.common_distributed import (  # noqa: E402
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)


class ProcessGroupNCCLTest(MultiProcessTestCase):
    def setUp(self):
        super(ProcessGroupNCCLTest, self).setUp()
        # NCCL_BLOCKING_WAIT overrides NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(find_free_port(8000, 8100)),
                "BAGUA_SERVICE_PORT": str(find_free_port(9000, 9100)),
            }
        )
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    def test_bagua_pg(self):
        # Need to use NCCL_BLOCKING_WAIT and not ASYNC_ERROR_HANDLING,
        # otherwise process will be taken down and we can't check for errors.
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ.update(
            {
                "WORLD_SIZE": str(self.world_size),
                "LOCAL_WORLD_SIZE": str(self.world_size),
                "RANK": str(self.rank),
                "LOCAL_RANK": str(self.rank),
            }
        )

        bagua.init_process_group()
        pg = c10d.new_group(ranks=list(range(0, self.world_size)))
        pg.bagua_patch()
        self.assertTrue(pg in bagua.communication._torch_to_bagua_pg_map)
        del pg
        c10d.destroy_process_group()
        self.assertEqual(len(bagua.communication._torch_to_bagua_pg_map), 0)


if __name__ == "__main__":
    unittest.main()
