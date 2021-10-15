from __future__ import print_function
import argparse
import torch
import torch.distributed as dist
import logging
import bagua.torch_api as bagua


def main():
    torch.set_printoptions(precision=20)
    parser = argparse.ArgumentParser(description="Communication Primitives Example")
    parser.parse_args()

    assert bagua.get_world_size() >= 2, "world size must be at least 2"

    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)

    comm = bagua.communication._get_default_group().get_global_communicator()

    # send, recv
    if bagua.get_rank() == 0:
        send_tensor = torch.rand(4, dtype=torch.float32).cuda()
        dist.send(send_tensor, 1)
        bagua.send(send_tensor, 1, comm=comm)
    elif bagua.get_rank() == 1:
        recv_tensor = torch.zeros(4, dtype=torch.float32).cuda()
        recv_tensor_bagua = torch.zeros(4, dtype=torch.float32).cuda()
        dist.recv(recv_tensor, 0)
        bagua.recv(recv_tensor_bagua, 0, comm=comm)
        assert torch.equal(
            recv_tensor, recv_tensor_bagua
        ), "recv_tensor:{a}, recv_tensor_bagua:{b}".format(
            a=recv_tensor, b=recv_tensor_bagua
        )

    # broadcast
    if bagua.get_rank() == 0:
        tensor = torch.rand(4, dtype=torch.float32).cuda()
        dist.broadcast(tensor, 0)
        bagua.broadcast(tensor, 0, comm=comm)
    else:
        recv_tensor = torch.zeros(4, dtype=torch.float32).cuda()
        recv_tensor_bagua = torch.zeros(4, dtype=torch.float32).cuda()
        dist.broadcast(recv_tensor, 0)
        bagua.broadcast(recv_tensor_bagua, 0, comm=comm)
        assert torch.equal(
            recv_tensor, recv_tensor_bagua
        ), "recv_tensor:{a}, recv_tensor_bagua:{b}".format(
            a=recv_tensor, b=recv_tensor_bagua
        )

    # allreduce
    send_tensor = torch.rand(4, dtype=torch.float32).cuda()
    send_tensor_bagua = torch.clone(send_tensor)
    recv_tensor_bagua = torch.zeros(4, dtype=torch.float32).cuda()
    dist.all_reduce(send_tensor)
    bagua.allreduce(send_tensor_bagua, recv_tensor_bagua, comm=comm)
    bagua.allreduce_inplace(send_tensor_bagua, comm=comm)
    assert torch.all(
        torch.isclose(send_tensor, recv_tensor_bagua)
    ), "send_tensor:{a}, recv_tensor_bagua:{b}".format(
        a=send_tensor, b=recv_tensor_bagua
    )
    assert torch.all(
        torch.isclose(send_tensor_bagua, recv_tensor_bagua)
    ), "send_tensor_bagua:{a}, recv_tensor_bagua:{b}".format(
        a=send_tensor_bagua, b=recv_tensor_bagua
    )

    # reduce
    dst = 1
    send_tensor = torch.rand(4, dtype=torch.float32).cuda()
    send_tensor_bagua = torch.clone(send_tensor)
    recv_tensor_bagua = torch.zeros(4, dtype=torch.float32).cuda()
    dist.reduce(send_tensor, dst)
    bagua.reduce(send_tensor_bagua, recv_tensor_bagua, dst=dst, comm=comm)
    bagua.reduce_inplace(send_tensor_bagua, dst=dst, comm=comm)
    assert torch.all(
        torch.isclose(send_tensor, send_tensor_bagua)
    ), "send_tensor:{a}, send_tensor_bagua:{b}".format(
        a=send_tensor, b=send_tensor_bagua
    )
    if bagua.get_rank() == dst:
        assert torch.all(
            torch.isclose(send_tensor_bagua, recv_tensor_bagua)
        ), "send_tensor_bagua:{a}, recv_tensor_bagua:{b}".format(
            a=send_tensor_bagua, b=recv_tensor_bagua
        )

    # allgather
    send_tensor = torch.rand(4, dtype=torch.float32).cuda()
    recv_tensor_bagua = torch.zeros(
        4 * bagua.get_world_size(), dtype=torch.float32
    ).cuda()
    recv_tensors = [
        torch.zeros(4, dtype=torch.float32).cuda()
        for i in range(bagua.get_world_size())
    ]
    dist.all_gather(recv_tensors, send_tensor)
    bagua.allgather(send_tensor, recv_tensor_bagua, comm=comm)
    assert torch.equal(
        torch.cat(recv_tensors), recv_tensor_bagua
    ), "recv_tensors:{a}, recv_tensor_bagua:{b}".format(
        a=recv_tensors, b=recv_tensor_bagua
    )

    # gather, scatter: Pytorch ProcessGroupNCCL does not support gather/scatter

    # reduce_scatter
    send_tensors = [
        torch.rand(4, dtype=torch.float32).cuda() for i in range(bagua.get_world_size())
    ]
    recv_tensor = torch.zeros(4, dtype=torch.float32).cuda()
    send_tensor_bagua = torch.cat(send_tensors)
    recv_tensor_bagua = torch.zeros(4, dtype=torch.float32).cuda()
    dist.reduce_scatter(recv_tensor, send_tensors)
    bagua.reduce_scatter(send_tensor_bagua, recv_tensor_bagua, comm=comm)
    assert torch.all(
        torch.isclose(recv_tensor, recv_tensor_bagua)
    ), "recv_tensor:{a}, recv_tensor_bagua:{b}".format(
        a=recv_tensor, b=recv_tensor_bagua
    )

    # alltoall
    send_tensors = [
        torch.rand(4, dtype=torch.float32).cuda() for i in range(bagua.get_world_size())
    ]
    recv_tensors = [
        torch.zeros(4, dtype=torch.float32).cuda()
        for i in range(bagua.get_world_size())
    ]
    recv_tensor_bagua = torch.zeros(
        4 * bagua.get_world_size(), dtype=torch.float32
    ).cuda()
    dist.all_to_all(recv_tensors, send_tensors)
    bagua.alltoall(torch.cat(send_tensors), recv_tensor_bagua, comm=comm)
    assert torch.equal(
        torch.cat(recv_tensors), recv_tensor_bagua
    ), "recv_tensors:{a}, recv_tensor_bagua:{b}".format(
        a=recv_tensors, b=recv_tensor_bagua
    )


if __name__ == "__main__":
    main()
