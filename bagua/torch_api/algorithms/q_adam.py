#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api import get_world_size
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm
from torch.optim.optimizer import Optimizer
import torch
import math
from typing import List, Tuple


class QAdamOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        warmup_steps: int = 100,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Create a dedicated optimizer used for
        `QAdam <https://bagua-tutorials.kwai-seattle.com/algorithms/q-adam>`_ algorithm.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate.
            warmup_steps: Number of steps to warm up by doing gradient allreduce before
                doing asynchronous model averaging. Use 0 to disable.
            betas: Coefficients used for computing running averages of gradient and its square.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(QAdamOptimizer, self).__init__(params, defaults)

        self.params_in_group = []
        self.exp_avgs_in_group = []
        self.step_id = 0
        self.warmup_steps = warmup_steps

        # initialize momentum and variance
        for group_id, group in enumerate(self.param_groups):
            params_with_grad = []
            exp_avgs = []
            for p in group["params"]:
                params_with_grad.append(p)
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                exp_avgs.append(state["exp_avg"])
            self.params_in_group.append(params_with_grad)
            self.exp_avgs_in_group.append(exp_avgs)

    def __setstate__(self, state):
        super(QAdamOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        self.step_id += 1
        for group_id, group in enumerate(self.param_groups):

            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for param_id, param in enumerate(group["params"]):
                state = self.state[param]

                if self.step_id < self.warmup_steps:
                    state["exp_avg"].mul_(beta1).add_(param.grad, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(
                        param.grad, param.grad, value=1 - beta2
                    )

                bias_correction1 = 1 - beta1 ** self.step_id
                bias_correction2 = 1 - beta2 ** self.step_id

                denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(
                    eps
                )
                step_size = lr / bias_correction1
                update = state["exp_avg"] / denom
                param.data.add_(-step_size * update)


class QAdamAlgorithm(Algorithm):
    def __init__(self, q_adam_optimizer: QAdamOptimizer, hierarchical: bool = True):
        """
        Create an instance of the
        `QAdam Algorithm <https://bagua-tutorials.kwai-seattle.com/algorithms/q-adam>`_
        .

        Args:
            q_adam_optimizer: A QAdamOptimizer initialized with model parameters.
            hierarchical: Enable hierarchical communication.
        """
        self.hierarchical = hierarchical
        self.optimizer = q_adam_optimizer
        self.warmup_steps = self.optimizer.warmup_steps

    def need_reset(self):
        if self.optimizer.step_id == self.warmup_steps:
            print(
                "QAdam starts to compress from step {}".format(self.optimizer.step_id)
            )
            return True
        else:
            return False

    def init_tensors(self, bagua_module: BaguaModule):
        parameters = bagua_module.bagua_build_params()

        for idx, (name, param) in enumerate(parameters.__reversed__()):
            param._q_adam_name = name
            param._q_adam_idx = idx

        tensor_groups = []
        for param_group, m_group in zip(
            self.optimizer.params_in_group, self.optimizer.exp_avgs_in_group
        ):
            for param, exp_avgs in zip(param_group, m_group):
                if self.optimizer.step_id < self.warmup_steps:
                    registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                        param._q_adam_name, bagua_module.bagua_module_name
                    )
                    param._q_adam_grad = registered_tensor
                    registered_tensor._q_adam_idx = param._q_adam_idx
                else:
                    registered_tensor = exp_avgs.ensure_bagua_tensor(
                        param._q_adam_name, bagua_module.bagua_module_name
                    )
                    registered_tensor._q_adam_grad = param.bagua_ensure_grad()
                    param._q_adam_momentum = registered_tensor
                    registered_tensor._q_adam_idx = param._q_adam_idx
                tensor_groups.append(registered_tensor)
        tensor_groups.sort(key=lambda x: x._q_adam_idx)
        return tensor_groups

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(
                bucket, flatten=True, name=str(idx), alignment=get_world_size()
            )
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        if self.optimizer.step_id < self.warmup_steps:
            bucket.append_centralized_synchronous_op(
                hierarchical=False,
                average=True,
                group=bagua_module._bagua_process_group,
            )
        else:

            def calculate_momentum(*args):
                beta1, beta2 = self.optimizer.param_groups[0]["betas"]
                for tensor in bucket.tensors:
                    tensor.mul_(beta1).add_(tensor._q_adam_grad, alpha=1 - beta1)

            bucket.append_python_op(calculate_momentum)
            bucket.append_centralized_synchronous_op(
                hierarchical=self.hierarchical,
                average=True,
                scattergather=True,
                compression="MinMaxUInt8",
                group=bagua_module._bagua_process_group,
            )

    def init_backward_hook(self, bagua_module: BaguaModule):
        def hook_momentum(parameter_name, parameter):
            parameter._q_adam_momentum.bagua_mark_communication_ready()

        def hook_grad(parameter_name, parameter):
            assert (
                parameter.grad.data_ptr() == parameter._q_adam_grad.data_ptr()
            ), "gradient data_ptr should match _q_adam_grad data_ptr"
            parameter._q_adam_grad.bagua_mark_communication_ready()

        return (
            hook_grad if self.optimizer.step_id < self.warmup_steps else hook_momentum
        )
