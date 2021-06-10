import bagua.torch_api as bagua
import torch
import logging
import argparse


logging.basicConfig(level=logging.DEBUG)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )

    args = parser.parse_args()

    # need to initialize torch.dist
    torch.cuda.set_device(args.local_rank)
    bagua.init_process_group()

    if args.rank == 0:
        tensors = [torch.rand(10).cuda() for _ in range(3)]
    else:
        tensors = [torch.zeros(10).cuda() for _ in range(3)]

    bagua.broadcast_coalesced(tensors, root=0)

    print("{}, after bcast, {}".format(bagua.get_rank(), tensors))

    tensor = torch.rand(10).cuda()
    bagua.allreduce(tensor)

    print("{}, after bcast, {}".format(bagua.get_rank(), tensor))
