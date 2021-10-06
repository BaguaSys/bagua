#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api import get_world_size
from bagua.torch_api.distributed import BaguaModule
from torch.optim.optimizer import Optimizer
import torch
import math
from typing import List, Tuple
from bagua.torch_api.algorithms_implementation.q_adam_implementation import (
    QAdamOptimizer_Implementation,
)


class QAdamOptimizer:
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

        self.params = params
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def reify(self) -> QAdamOptimizer_Implementation:
        return QAdamOptimizer_Implementation(
            params=self.params,
            lr=self.lr,
            warmup_steps=self.warmup_steps,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
