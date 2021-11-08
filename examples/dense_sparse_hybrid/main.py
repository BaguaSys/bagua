import bagua.torch_api as bagua
from bagua.torch_api.bucket import BaguaBucket
from argparse import ArgumentParser
import torch
from mpi4py import MPI
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--master_addr", type=str, required=True)
    parser.add_argument("--master_port", type=int, required=True)
    parser.add_argument("--bagua_service_port", type=int, required=True)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    os.environ["WORLD_SIZE"] = str(comm.Get_size())
    os.environ["LOCAL_WORLD_SIZE"] = str("1")
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["BAGUA_SERVICE_PORT"] = str(args.bagua_service_port)
    os.environ["RANK"] = str(comm.Get_rank())
    os.environ["LOCAL_RANK"] = str("0")

    bagua.init_process_group()
    bagua_backend = bagua.get_backend("demo")

    dense_list = [
        torch.ones(
            [1024, 4], device="cuda:{}".format(bagua.get_local_rank())
        ).ensure_bagua_tensor(
            name="tensor-{}".format(i),
            module_name="demo",
        )
        for i in range(20)
    ]
    bagua_bucket = BaguaBucket(dense_list, "dense", True)
    bagua_backend.register_ordered_buckets([bagua_bucket.backend_bucket])
    bagua_bucket.append_centralized_synchronous_op()

    benchmark_loop = 1000
    for i in range(benchmark_loop):
        print('i={}'.format(i))
        # timer
        for t in dense_list:
            t.bagua_mark_communication_ready()
        bagua_backend.wait_pending_comm_ops()
