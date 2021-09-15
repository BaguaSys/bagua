import os
import unittest
import multiprocessing

import bagua.torch_api as bagua
from bagua.torch_api.contrib.sync_batchnorm import SyncBatchNorm

import torch

from tests.internal.common_utils import find_free_port


class Result_Forward(object):
    def __init__(self):
        bn_layer = SyncBatchNorm(num_features=8)
        self.bn_output = torch.zeros(32, 8, 16, 16).cuda().float()
        self.running_mean = torch.empty_like(bn_layer.running_mean)
        self.running_var = torch.empty_like(bn_layer.running_var)


class Result_Backward(object):
    def __init__(self):
        bn_layer = SyncBatchNorm(num_features=8)
        self.weight_grad = torch.empty_like(bn_layer.weight.data)
        self.bias_grad = torch.empty_like(bn_layer.bias.data)
        self.input_grad = torch.zeros(32, 8, 16, 16).cuda().float()


def _init_torch_env(rank, nprocs, env):
    # set deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(rank)

    # init torch distributed process group
    torch.cuda.set_device(rank)
    init_method = "tcp://" + env["MASTER_ADDR"] + ":" + env["MASTER_PORT"]
    torch.distributed.init_process_group(
        world_size=nprocs,
        rank=rank,
        backend="nccl",
        init_method=init_method,
    )


def _init_bagua_env(rank, env):
    # set deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(rank)
    # initialize subprocess env
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


def run_torch_sync_bn(rank, nprocs, torch_results_forward, results_backward, envs):
    _init_torch_env(rank, nprocs, envs)

    input_d = torch.randn(32, 8, 16, 16).cuda().float()
    input_d.requires_grad_()
    sync_bn_ddp = torch.nn.SyncBatchNorm(num_features=8)

    sync_bn_ddp.cuda(rank)
    input_d = input_d.cuda(rank)
    sync_bn_ddp_out = sync_bn_ddp(input_d)

    ddp_ret1 = torch_results_forward[rank]
    ddp_ret1.bn_output.copy_(sync_bn_ddp_out)
    ddp_ret1.running_mean.copy_(sync_bn_ddp.running_mean)
    ddp_ret1.running_var.copy_(sync_bn_ddp.running_var)

    sync_bn_ddp_out.sum().backward()

    ddp_ret2 = results_backward[rank]
    ddp_ret2.weight_grad.copy_(sync_bn_ddp.weight.grad)
    ddp_ret2.bias_grad.copy_(sync_bn_ddp.bias.grad)
    ddp_ret2.input_grad.copy_(input_d.grad)


def run_bagua_sync_bn(rank, nprocs, bagua_results_forward, results_backward, envs):
    _init_bagua_env(rank, envs)

    input_b = torch.randn(32, 8, 16, 16).cuda().float()
    input_b.requires_grad_()
    sync_bn_bagua = SyncBatchNorm(num_features=8)

    sync_bn_bagua.cuda(rank)
    input_b = input_b.cuda(rank)
    sync_bn_bagua_out = sync_bn_bagua(input_b)

    bagua_ret1 = bagua_results_forward[rank]
    bagua_ret1.bn_output.copy_(sync_bn_bagua_out)
    bagua_ret1.running_mean.copy_(sync_bn_bagua.running_mean)
    bagua_ret1.running_var.copy_(sync_bn_bagua.running_var)

    sync_bn_bagua_out.sum().backward()

    bagua_ret2 = results_backward[rank]
    bagua_ret2.weight_grad.copy_(sync_bn_bagua.weight.grad)
    bagua_ret2.bias_grad.copy_(sync_bn_bagua.bias.grad)
    bagua_ret2.input_grad.copy_(input_b.grad)


class Test_Sync_Bn(unittest.TestCase):
    def test_syncbn(self):
        nprocs = torch.cuda.device_count()
        print(nprocs)
        env = {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(find_free_port(8000, 8100)),
        }

        mp = multiprocessing.get_context("spawn")
        torch_results_forward = [Result_Forward() for _ in range(nprocs)]
        torch_results_backward = [Result_Backward() for _ in range(nprocs)]
        processes = []

        for i in range(nprocs):
            p = mp.Process(
                target=run_torch_sync_bn,
                args=(
                    i,
                    nprocs,
                    torch_results_forward,
                    torch_results_backward,
                    env,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.assertTrue(p.exitcode == 0)

        env = {
            "WORLD_SIZE": str(nprocs),
            "LOCAL_WORLD_SIZE": str(nprocs),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(find_free_port(8000, 8100)),
            "BAGUA_SERVICE_PORT": str(find_free_port(9000, 9100)),
        }

        bagua_results_forward = [Result_Forward() for _ in range(nprocs)]
        bagua_results_backward = [Result_Backward() for _ in range(nprocs)]
        processes = []
        for i in range(nprocs):
            p = mp.Process(
                target=run_bagua_sync_bn,
                args=(
                    i,
                    nprocs,
                    bagua_results_forward,
                    bagua_results_backward,
                    env,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            self.assertTrue(p.exitcode == 0)

        for rank in range(nprocs):
            self.assertTrue(
                torch.all(
                    torch.isclose(
                        torch_results_forward[rank].bn_output,
                        bagua_results_forward[rank].bn_output,
                    )
                ).item()
            )
            self.assertTrue(
                torch.all(
                    torch.isclose(
                        torch_results_forward[rank].running_mean,
                        bagua_results_forward[rank].running_mean,
                    )
                ).item()
            )
            self.assertTrue(
                torch.all(
                    torch.isclose(
                        torch_results_forward[rank].running_var,
                        bagua_results_forward[rank].running_var,
                    )
                ).item()
            )

            self.assertTrue(
                torch.all(
                    torch.isclose(
                        torch_results_backward[rank].weight_grad,
                        bagua_results_backward[rank].weight_grad,
                    )
                ).item()
            )
            self.assertTrue(
                torch.all(
                    torch.isclose(
                        torch_results_backward[rank].bias_grad,
                        bagua_results_backward[rank].bias_grad,
                    )
                ).item()
            )
            self.assertTrue(
                torch.all(
                    torch.isclose(
                        torch_results_backward[rank].input_grad,
                        bagua_results_backward[rank].input_grad,
                    )
                ).item()
            )


if __name__ == "__main__":
    unittest.main()
