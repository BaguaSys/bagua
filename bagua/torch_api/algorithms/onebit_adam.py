#!/usr/bin/env python3

from bagua.torch_api.globals import _get_global_state
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api import get_world_size
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm
from torch.optim.optimizer import Optimizer
import torch
import math
from typing import List


class OnebitAdamAlgorithm(Algorithm):
    def __init__(
        self,
        onebit_optimizer: Optimizer,
        warmup_steps: int,
        hierarchical_reduce: bool = True,
    ):

        self.warmup_steps = warmup_steps
        self.hierarchical_reduce = hierarchical_reduce
        self.optimizer = onebit_optimizer

    def need_reset(self):

        if self.optimizer.step_id == self.warmup_steps:
            print(
                "Onebit Adam starts to compress from step {}".format(
                    self.optimizer.step_id
                )
            )
            return True
        else:
            return False

    def init_tensors(self, bagua_module: BaguaModule):

        parameters = bagua_module.bagua_build_params()

        for name, param in parameters:
            param._one_bit_name = name

        tensors = []
        for param_group, m_group in zip(
            self.optimizer.params_in_group, self.optimizer.exp_avgs_in_group
        ):
            for param, exp_avgs in zip(param_group, m_group):
                if self.optimizer.step_id < self.warmup_steps:
                    registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                        param._one_bit_name
                    )
                    param._one_bit_grad = registered_tensor
                else:
                    registered_tensor = exp_avgs.ensure_bagua_tensor(
                        param._one_bit_name
                    )
                    registered_tensor._one_bit_grad = param.bagua_ensure_grad()
                    param._one_bit_momentum = registered_tensor
                tensors.append(registered_tensor)

        return tensors

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        """
        Given the bucketing suggestion from Bagua, return the actual Bagua buckets.
        The default implementation follows the suggestion to do the bucketing.

        Args:
            tensors: Bagua tensors grouped in different
                lists, representing Bagua's suggestion on how to bucketing the
                tensors.

        Returns:
            A list of Bagua buckets.
        """
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
        bucket.backend_bucket.clear_ops()
        if self.optimizer.step_id < self.warmup_steps:
            bucket.backend_bucket.append_centralized_synchronous_op(
                bagua_module.bagua_global_communicator,
                bagua_module.bagua_intra_node_communicator,
                hierarchical=False,
                average=True,
            )
        else:
            def calculate_momentum(*args):
                with torch.cuda.stream(_get_global_state().get_communication_stream()):
                    beta1, beta2 = self.optimizer.param_groups[0]["betas"]
                    for tensor in bucket.tensors:
                        tensor.mul_(beta1).add_(tensor._one_bit_grad, alpha=1 - beta1)

            bucket.backend_bucket.append_python_op(calculate_momentum)
            bucket.backend_bucket.append_centralized_synchronous_op(
                bagua_module.bagua_inter_node_communicator,
                bagua_module.bagua_intra_node_communicator,
                hierarchical=self.hierarchical_reduce,
                average=True,
                scattergather=True,
                compression="MinMaxUInt8",
            )

    def init_backward_hook(self, bagua_module: BaguaModule):
        def hook_momentum(parameter_name, parameter):
            parameter._one_bit_momentum.bagua_mark_communication_ready()

        def hook_grad(parameter_name, parameter):
            assert (
                parameter._one_bit_grad.data_ptr() == parameter.grad.data_ptr()
            ), "one bit grad data_ptr should match grad data_ptr"
            parameter._one_bit_grad.bagua_mark_communication_ready()

        return (
            hook_grad if self.optimizer.step_id < self.warmup_steps else hook_momentum
        )


class OnebitAdamOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        warmup_steps=100,
        is_bert=False,
        freeze_test_step=-1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            warmup_steps=warmup_steps,
            is_bert=is_bert,
            freeze_test_step=freeze_test_step,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(OnebitAdamOptimizer, self).__init__(params, defaults)

        self.params_in_group = []
        self.exp_avgs_in_group = []
        self.step_id = 0

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

    def step(self, closure=None):
        # Here we assume grad or state["exp_avg"] have already been updated and averaged.
        # This step only updates weights.
        self.step_id += 1
        for group_id, group in enumerate(self.param_groups):

            lr = group["lr"]
            # weight_decay = group["weight_decay"] # TODO: @ganshaoduo check if this should be used
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for param_id, param in enumerate(group["params"]):
                state = self.state[param]

                if self.step_id < group["warmup_steps"]:
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
