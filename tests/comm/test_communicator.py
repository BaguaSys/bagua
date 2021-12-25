import unittest
import torch
import os
from bagua.torch_api.communication import (
    _get_default_group,
    allreduce,
    allreduce_inplace,
    send,
    recv,
    allgather,
    barrier,
    broadcast_object,
    reduce,
    reduce_inplace,
    reduce_scatter,
    reduce_scatter_inplace,
    ReduceOp,
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
        comm.allreduce_inplace(
            data.ensure_bagua_tensor().bagua_backend_tensor(), ReduceOp.AVG
        )

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


def run_bcastobject(rank, nprocs, results, env):
    init_env(rank, env)

    if rank == 0:
        state_dict = {"lr": 0.02, "weight_decay": 1e-4, "momentum": 0.9}
    else:
        state_dict = {}

    state_dict = broadcast_object(state_dict, 0)

    ret = True
    for i in range(nprocs):
        ret = (
            state_dict["lr"] == 0.02
            and state_dict["weight_decay"] == 1e-4
            and state_dict["momentum"] == 0.9
        )

    results[rank].ret[0] = ret


def run_avg(rank, nprocs, results, env):
    init_env(rank, env)

    def reduce_fn(send_tensor, recv_tensor, op):
        reduce(send_tensor, recv_tensor, 0, op)

    def reduce_inplace_fn(tensor, op):
        reduce_inplace(tensor, 0, op)

    fns = [reduce_fn, allreduce]
    inplace_fns = [reduce_inplace_fn, allreduce_inplace]

    succ = True
    for fn in fns:
        send_tensor = torch.rand(100).cuda()
        recv_tensor = torch.zeros_like(send_tensor)

        send_tensor_clone = send_tensor.clone().detach()
        recv_tensor_clone = recv_tensor.clone().detach()

        fn(send_tensor, recv_tensor, op=ReduceOp.AVG)
        fn(send_tensor_clone, recv_tensor_clone, op=ReduceOp.SUM)

        recv_tensor_clone /= nprocs

        torch.cuda.synchronize()
        succ = succ and torch.equal(recv_tensor, recv_tensor_clone)

    for fn in inplace_fns:
        tensor = torch.rand(100).cuda()
        tensor_clone = tensor.clone().detach()

        fn(tensor, op=ReduceOp.AVG)
        fn(tensor_clone, op=ReduceOp.SUM)

        tensor_clone /= nprocs

        torch.cuda.synchronize()
        succ = succ and torch.equal(tensor, tensor_clone)

    results[rank].ret[0] = succ


def run_reduce_scatter(rank, nprocs, results, env):
    init_env(rank, env)

    send_tensor = torch.rand(100 * nprocs).cuda()
    recv_tensor = torch.rand(100).cuda()

    send_tensor_clone = send_tensor.clone().detach()
    recv_tensor_clone = recv_tensor.clone().detach()

    reduce_scatter(send_tensor, recv_tensor, op=ReduceOp.AVG)
    reduce_scatter(send_tensor_clone, recv_tensor_clone, op=ReduceOp.SUM)

    recv_tensor_clone /= nprocs

    tensor = torch.rand(100 * nprocs).cuda()
    tensor_clone = tensor.clone().detach()

    reduce_scatter_inplace(tensor, op=ReduceOp.AVG)
    reduce_scatter_inplace(tensor_clone, op=ReduceOp.AVG)

    tensor_clone /= nprocs

    torch.cuda.synchronize()

    results[rank].ret[0] = torch.equal(recv_tensor, recv_tensor_clone) and torch.equal(
        tensor, tensor_clone
    )


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

    @skip_if_cuda_not_available()
    def test_bcastobject(self):
        self.run_test_locally(run_bcastobject)

    @skip_if_cuda_not_available()
    def test_avg(self):
        self.run_test_locally(run_avg)

    @skip_if_cuda_not_available()
    def test_reduce_scatter(self):
        self.run_test_locally(run_reduce_scatter)


if __name__ == "__main__":
    unittest.main()
