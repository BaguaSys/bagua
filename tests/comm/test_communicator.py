import unittest
import torch
import os
from bagua.torch_api.communication import init_bagua_communicator
from tests.internal.common_utils import find_free_port
import torch.multiprocessing as mp
import bagua.torch_api as bagua
import threading
import time


def run_abort(rank, nprocs):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # init bagua distributed process group
    torch.cuda.set_device(rank)
    bagua.init_process_group()

    comm_stream = torch.cuda.Stream()
    comm = init_bagua_communicator(model_name="test", stream=comm_stream)

    def abort():
        time.sleep(2)
        comm.abort()

    threading.Thread(target=abort).start()

    for i in range(rank % 2 + 1):
        data = torch.rand(10).cuda()
        comm.allreduce_inplace(data.to_bagua_tensor().bagua_backend_tensor(), 10)


def run_test_locally(fn):
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
        fn,
        nprocs=nprocs,
        args=(nprocs,),
    )


class TestCommunication(unittest.TestCase):
    def test_abort(self):
        run_test_locally(run_abort)


if __name__ == "__main__":
    unittest.main()
