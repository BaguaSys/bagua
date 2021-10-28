import unittest
import torch
import os
from bagua.torch_api.communication import (
    _get_default_group,
    allreduce,
    send,
    recv,
    allgather,
    barrier,
)
from tests.internal.common_utils import find_free_port
import multiprocessing
import bagua.torch_api as bagua
import threading
import time
from tests import skip_if_cuda_not_available


class Result(object):
    def __init__(self):
        self.ret = torch.Tensor([False]).bool()
        self.data = torch.Tensor([0.0])


def init_env(rank, env):
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


def run_abort(rank, nprocs, results, env):
    init_env(rank, env)

    os.environ["NCCL_PROTO"] = "^LL128"

    comm_stream = torch.cuda.Stream()
    comm = _get_default_group().get_global_communicator()

    def abort():
        time.sleep(10)
        comm.abort()

    threading.Thread(target=abort).start()

    with torch.cuda.stream(comm_stream):
        data = torch.rand(10).cuda()

    for _ in range(rank + 1):
        comm.allreduce_inplace(data.to_bagua_tensor().bagua_backend_tensor(), 10)

    comm_stream.synchronize()


def run_allreduce(rank, nprocs, results, env):
    init_env(rank, env)

    send_tensor = torch.rand(100).cuda()
    recv_tensor = torch.zeros_like(send_tensor)

    tensor = send_tensor.clone()

    allreduce(send_tensor, recv_tensor)

    torch.distributed.all_reduce(tensor)

    results[rank].ret[0] = torch.equal(recv_tensor, tensor)


def run_p2p(rank, nprocs, results, env):
    init_env(rank, env)

    send_tensor = torch.rand(100).cuda()
    recv_tensor = torch.zeros_like(send_tensor)

    if rank % 2 == 0:
        send(send_tensor, dst=(rank + 1) % nprocs)
        results[rank].data.copy_(torch.norm(send_tensor))
    else:
        recv(recv_tensor, src=(rank - 1 + nprocs) % nprocs)
        results[rank].data.copy_(torch.norm(recv_tensor))


def run_allgather(rank, nprocs, results, env):
    init_env(rank, env)

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


def run_barrier(rank, nprocs, results, env):
    init_env(rank, env)
    barrier()


class TestCommunication(unittest.TestCase):
    def run_test_locally(self, fn):
        nprocs = torch.cuda.device_count()
        env = {
            "WORLD_SIZE": str(nprocs),
            "LOCAL_WORLD_SIZE": str(nprocs),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(find_free_port(8000, 8100)),
            "BAGUA_SERVICE_PORT": str(find_free_port(9000, 9100)),
        }

        mp = multiprocessing.get_context("spawn")
        results = [Result() for _ in range(nprocs)]
        processes = []
        for i in range(nprocs):
            p = mp.Process(
                target=fn,
                args=(i, nprocs, results, env),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=60)
            self.assertTrue(p.exitcode == 0)

        return results

    @skip_if_cuda_not_available()
    def test_abort(self):
        self.run_test_locally(run_abort)

    @skip_if_cuda_not_available()
    def test_allreduce(self):
        results = self.run_test_locally(run_allreduce)
        for ret in results:
            self.assertTrue(ret.ret.item())

    @skip_if_cuda_not_available()
    def test_p2p(self):
        results = self.run_test_locally(run_p2p)

        i = 1
        while i < len(results):
            self.assertTrue(torch.equal(results[i].data, results[i - 1].data))
            i += 2

    @skip_if_cuda_not_available()
    def test_allgather(self):
        results = self.run_test_locally(run_allgather)
        for ret in results:
            self.assertTrue(ret.ret.item())

    @skip_if_cuda_not_available()
    def test_barrier(self):
        self.run_test_locally(run_barrier)


if __name__ == "__main__":
    unittest.main()
