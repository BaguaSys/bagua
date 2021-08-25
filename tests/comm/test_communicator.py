import unittest
import torch
import os
from bagua.torch_api.communication import (
    init_bagua_communicator,
    allreduce,
    send,
    recv,
    allgather,
)
from tests.internal.common_utils import find_free_port
import torch.multiprocessing as mp
import bagua.torch_api as bagua
import threading
import time


class Result(object):
    def __init__(self):
        self.ret = torch.Tensor([False]).bool()
        self.data = torch.Tensor([0.0])


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
        time.sleep(10)
        comm.abort()

    threading.Thread(target=abort).start()

    with torch.cuda.stream(comm_stream):
        data = torch.rand(10).cuda()

    for _ in range(rank + 1):
        comm.allreduce_inplace(data.to_bagua_tensor().bagua_backend_tensor(), 10)

    comm_stream.synchronize()


def run_allreduce(rank, nprocs, results):
    init_env(rank)

    send_tensor = torch.rand(100).cuda()
    recv_tensor = torch.zeros_like(send_tensor)

    tensor = send_tensor.clone()

    allreduce(send_tensor, recv_tensor)

    torch.distributed.all_reduce(tensor)

    results[rank].ret[0] = torch.equal(recv_tensor, tensor)


def run_p2p(rank, nprocs, results):
    init_env(rank)

    send_tensor = torch.rand(100).cuda()
    recv_tensor = torch.zeros_like(send_tensor)

    if rank % 2 == 0:
        send(send_tensor, dst=(rank + 1) % nprocs)
        results[rank].data.copy_(torch.norm(send_tensor))
    else:
        recv(recv_tensor, src=(rank - 1 + nprocs) % nprocs)
        results[rank].data.copy_(torch.norm(recv_tensor))


def run_allgather(rank, nprocs, results):
    init_env(rank)

    send_tensor = torch.rand(100).cuda()
    recv_tensor = torch.zeros(
        [nprocs, 100], device=send_tensor.device, dtype=send_tensor.dtype
    )

    tensor = send_tensor.clone()
    tensor_list = [torch.zeros_like(tensor) for _ in range(nprocs)]

    allgather(send_tensor, recv_tensor)

    torch.distributed.all_gather(tensor_list, tensor)

    ret = True
    for i in range(nprocs):
        ret = ret and torch.equal(recv_tensor[i], tensor_list[i])

    results[rank].ret[0] = ret


def run_test_locally(fn):
    if not torch.cuda.is_available():
        print("skip tests since cuda is not available")
        return []

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

    def test_p2p(self):
        results = run_test_locally(run_p2p)

        i = 1
        while i < len(results):
            self.assertTrue(torch.equal(results[i].data, results[i - 1].data))
            i += 2

    def test_allgather(self):
        results = run_test_locally(run_allgather)
        for ret in results:
            self.assertTrue(ret.ret.item())


if __name__ == "__main__":
    unittest.main()
