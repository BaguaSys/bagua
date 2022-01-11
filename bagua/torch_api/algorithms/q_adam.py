#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
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
        `QAdam <https://tutorials.baguasys.com/algorithms/q-adam>`_ algorithm.

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
        if warmup_steps <= 0:
            raise ValueError(
                "Invalid warmup_steps parameter, must be larger than 0: {}".format(
                    warmup_steps
                )
            )

        super(QAdamOptimizer, self).__init__(params, defaults)
        self.warmup_steps = warmup_steps

    def __setstate__(self, state):
        super(QAdamOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_id, group in enumerate(self.param_groups):

            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for param_id, param in enumerate(group["params"]):
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step_id = state["step"]

                grad = param.grad
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                if step_id < self.warmup_steps:
                    state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(
                        grad, grad, value=1 - beta2
                    )

                bias_correction1 = 1 - beta1 ** step_id
                bias_correction2 = 1 - beta2 ** step_id

                denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(
                    eps
                )
                step_size = lr / bias_correction1
                update = state["exp_avg"] / denom
                param.data.add_(-step_size * update)

        return loss


class QAdamAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        q_adam_optimizer: QAdamOptimizer,
        hierarchical: bool = True,
    ):
        """
        Implementation of the
        `QAdam Algorithm <https://tutorials.baguasys.com/algorithms/q-adam>`_
        .

        Args:
            process_group: The process group to work on.
            q_adam_optimizer: A QAdamOptimizer initialized with model parameters.
            hierarchical: Enable hierarchical communication.
        """
        super(QAdamAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.optimizer = q_adam_optimizer
        self.warmup_steps = self.optimizer.warmup_steps

    @property
    def optimizer_step_id(self):
        param = self.optimizer.param_groups[0]["params"][0]
        return self.optimizer.state[param].get("step", 0)

    def need_reset(self):
        if self.optimizer_step_id == self.warmup_steps:
            print(
                "QAdam starts to compress from step {}".format(self.optimizer_step_id)
            )
            return True
        else:
            return False

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel):
        parameters = bagua_ddp.bagua_build_params()

        for idx, (name, param) in enumerate(parameters.__reversed__()):
            param._q_adam_name = name
            param._q_adam_idx = idx

        tensor_groups = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if self.optimizer_step_id < self.warmup_steps:
                    # register grad
                    registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                        param._q_adam_name,
                        bagua_ddp.bagua_module_name,
                        getter_closure=lambda param: param.grad,
                        setter_closure=lambda param, t: setattr(param, "grad", t),
                    )
                else:
                    # register first momentum
                    def set_momentum_fn(param, t):
                        self.optimizer.state[param]["exp_avg"] = t

                    registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
                        param._q_adam_name,
                        bagua_ddp.bagua_module_name,
                        getter_closure=lambda param: self.optimizer.state[param][
                            "exp_avg"
                        ],
                        setter_closure=set_momentum_fn,
                    )

                tensor_groups.append(registered_tensor)
        tensor_groups.sort(key=lambda x: x._q_adam_idx)
        return tensor_groups

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(
                bucket,
                flatten=do_flatten,
                name=str(idx),
                alignment=self.process_group.get_global_communicator().nranks(),
            )
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        if self.optimizer_step_id < self.warmup_steps:
            bucket.append_centralized_synchronous_op(
                hierarchical=False,
                average=True,
                group=self.process_group,
            )
        else:

            def calculate_momentum(*args):
                beta1, beta2 = self.optimizer.param_groups[0]["betas"]
                for tensor in bucket.tensors:
                    tensor.bagua_getter_closure().mul_(beta1).add_(
                        tensor.grad, alpha=1 - beta1
                    )

            bucket.append_python_op(calculate_momentum, group=self.process_group)
            bucket.append_centralized_synchronous_op(
                hierarchical=self.hierarchical,
                average=True,
                scattergather=True,
                compression="MinMaxUInt8",
                group=self.process_group,
            )

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook_momentum(parameter_name, parameter):
            assert (
                parameter.bagua_backend_tensor().data_ptr()
                == self.optimizer.state[parameter]["exp_avg"].data_ptr()
            ), "bagua backend tensor data_ptr should match _q_adam_momentum data_ptr"
            parameter.bagua_mark_communication_ready()

        def hook_grad(parameter_name, parameter):
            assert (
                parameter.bagua_backend_tensor().data_ptr() == parameter.grad.data_ptr()
            ), "bagua backend tensor data_ptr should match _q_adam_grad data_ptr"
            parameter.bagua_mark_communication_ready()

        return (
            hook_grad if self.optimizer_step_id < self.warmup_steps else hook_momentum
        )


class QAdamAlgorithm(Algorithm):
    def __init__(self, q_adam_optimizer: QAdamOptimizer, hierarchical: bool = True):
        """
        Create an instance of the
        `QAdam Algorithm <https://tutorials.baguasys.com/algorithms/q-adam>`_
        .

        Args:
            q_adam_optimizer: A QAdamOptimizer initialized with model parameters.
            hierarchical: Enable hierarchical communication.
        """
        self.hierarchical = hierarchical
        self.optimizer = q_adam_optimizer

    def reify(self, process_group: BaguaProcessGroup) -> QAdamAlgorithmImpl:
        return QAdamAlgorithmImpl(
            process_group,
            q_adam_optimizer=self.optimizer,
            hierarchical=self.hierarchical,
        )
