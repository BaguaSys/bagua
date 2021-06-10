from bagua.torch_api.distributed import init_communicator
import cupy
import torch
import argparse
import os
import logging

"""
run the script with the following commands:
python3 -m torch.distributed.launch --nproc_per_node=8 p2p.py
"""
logging.basicConfig(level=logging.DEBUG)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )

    args = parser.parse_args()
    # need to initialize torch.dist
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    comm = init_communicator("nccl")

    torch.cuda.set_device(comm.get_rank())

    print("rank: {}, size: {}".format(comm.get_rank(), comm.get_size()))

    sendbuf = cupy.random.random(10)
    recvbuf = cupy.zeros([comm.get_size(), sendbuf.size], dtype=sendbuf.dtype)

    sendbuf2 = cupy.arange(10)
    recvbuf2 = cupy.zeros([comm.get_size(), sendbuf2.size], dtype=sendbuf2.dtype)

    comm.gather(sendbuf, recvbuf, 0)
    comm.gather(sendbuf2, recvbuf2, 1)

    print("{} after send recv, {}, {}".format(comm.get_rank(), recvbuf, recvbuf2))
