import time
import argparse

import cupy
import torch
import torch.distributed as dist

from bagua.torch_api.distributed import init_sub_communicator
from bagua.torch_api.utils import cupy2torch

# import sys
# sys.path.append("../")
# sys.path.append("../../")
# from tests.nccl.communicators import init_sub_communicator, torch2cupy, cupy2torch


def run(args):
    """Distributed Synchronous SGD Example"""
    device = torch.device(
        "cuda:{}".format(args.gpu_id)
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )
    print("device={}".format(device))
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.rank)

    round = 10
    comm_ranks = [1, 2, 3]
    sg_ranks = [0, 1, 2]  # rank in subgroup
    sg_size = len(sg_ranks)

    comm = None
    if args.rank in comm_ranks:
        index = comm_ranks.index(args.rank)
        sg_rank = sg_ranks[index]
        comm = init_sub_communicator("nccl", sg_rank, sg_size)
        print("subgroup rank: {}, size: {}".format(sg_rank, sg_size))

    send_size = 100

    for i in range(round):
        if args.rank in comm_ranks:
            torch_tensor = torch.full(
                (sg_size, send_size), args.rank, dtype=torch.float64
            )
            # print("torch tensor shape = {}".format(torch_tensor.shape))
            # cupy_tensor = torch2cupy(torch_tensor).reshape((sg_size-1, send_size))
            sendbuf = cupy.asarray(torch_tensor.numpy())
            # print("sendbuf shape = {}".format(sendbuf.shape))
            recvbuf = cupy.zeros((sg_size, send_size), dtype=cupy.float64)
            comm_start = time.time()
            comm.scatter_all(sendbuf, recvbuf, send_size)
            comm_end = time.time()
            print("communication cost {} s".format(comm_end - comm_start))
            torch_tensor = cupy2torch(recvbuf)
            print(
                "tensor after communication {} = {}".format(
                    torch_tensor.shape, torch_tensor
                )
            )
        dist.barrier()
        print(">>>>>>>>>>> round {} finishes".format(i))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", type=str, default="nccl", help="Name of the backend to use."
    )
    parser.add_argument(
        "-i",
        "--init-method",
        type=str,
        default="env://",
        help="URL specifying how to initialize the package.",
    )
    parser.add_argument(
        "-s",
        "--world-size",
        type=int,
        default=1,
        help="Number of processes participating in the job.",
    )
    parser.add_argument(
        "-r", "--rank", type=int, default=0, help="Rank of the current process."
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=1)
    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        dist.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
        )

    run(args)


if __name__ == "__main__":
    main()
