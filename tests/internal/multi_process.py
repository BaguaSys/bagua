import unittest
from tests.internal.common_utils import find_free_port
import multiprocessing
import os
import torch
import bagua.torch_api as bagua


class MultiProcessTestCase(unittest.TestCase):
    def run_test_locally(self, fn, nprocs, args, results):
        env = {
            "WORLD_SIZE": str(nprocs),
            "LOCAL_WORLD_SIZE": str(nprocs),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(find_free_port(8000, 8100)),
            "BAGUA_SERVICE_PORT": str(find_free_port(9000, 9100)),
        }

        mp = multiprocessing.get_context("spawn")
        processes = []
        for i in range(nprocs):
            p = mp.Process(
                target=fn,
                args=(
                    i,
                    nprocs,
                    args,
                    results,
                    env,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=60)
            self.assertTrue(p.exitcode == 0)


def setup_bagua_env(rank, env):
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
