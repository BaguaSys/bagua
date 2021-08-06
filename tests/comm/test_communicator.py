import unittest
import torch
import os
from bagua.torch_api.communication import init_bagua_communicator, allreduce
from tests.internal.common_utils import find_free_port
import torch.multiprocessing as mp
import bagua.torch_api as bagua
import threading
import time


class Result(object):
    def __init__(self):
        self.ret = torch.Tensor([False]).bool()


def init_env(rank):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # init bagua distributed process group
    torch.cuda.set_device(rank)
    bagua.init_process_group()


def run_abort(rank, nprocs, results):
    init_env(rank)

    comm_stream = torch.cuda.Stream()
    comm = init_bagua_communicator(model_name="test_comm", stream=comm_stream)

    def abort():
        time.sleep(2)
        comm.abort()

    threading.Thread(target=abort).start()

    for _ in range(rank + 1):
        data = torch.rand(10).cuda()
        comm.allreduce_inplace(data.to_bagua_tensor().bagua_backend_tensor(), 10)


def run_allreduce(rank, nprocs, results):
    init_env(rank)

    send_tensor = torch.rand(100).cuda()
    recv_tensor = torch.zeros_like(send_tensor)

    tensor = send_tensor.clone()

    allreduce(send_tensor, recv_tensor)

    torch.distributed.all_reduce(tensor)

    results[rank].ret[0] = torch.equal(recv_tensor, tensor)


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

    results = [Result() for _ in range(nprocs)]
    mp.spawn(
        fn,
        nprocs=nprocs,
        args=(nprocs, results),
    )

    return results


class TestCommunication(unittest.TestCase):
    def test_abort(self):
        run_test_locally(run_abort)

    def test_allreduce(self):
        results = run_test_locally(run_allreduce)
        for ret in results:
            self.assertTrue(ret.ret.item())


if __name__ == "__main__":
    unittest.main()
