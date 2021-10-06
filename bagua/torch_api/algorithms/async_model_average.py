#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.distributed import BaguaModule
from typing import List
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.env import get_rank
from enum import IntEnum
import threading
import time
import torch
import logging
from bagua.torch_api.algorithms_implementation.async_model_average_implementation import (
    AsyncModelAverageAlgorithm_Implementation,
)

__all__ = ["AsyncModelAverageAlgorithm"]


class AsyncModelAverageAlgorithm:
    def __init__(
        self,
        peer_selection_mode: str = "all",
        sync_interval_ms: int = 500,
        warmup_steps: int = 0,
    ):
        """
        Create an instance of the
        `AsyncModelAverage <https://bagua-tutorials.kwai-seattle.com/algorithms/async-model-average.html>`_
        algorithm.

        The asynchronous implementation is experimental, and imposes some restrictions.
        With such asynchronous algorithm, the number of iterations on each worker are different. Therefore
        the current implementation assumes that the dataset is an endless stream, and all workers continuously
        synchronize between each other.

        Users should call :meth:`abort` to manually stop the algorithm's continuous synchronization process.

        Args:
            peer_selection_mode (str): The way how workers communicate with each other. Currently ``"all"`` is supported.
                ``"all"`` means all workers' weights are synchronized during each communication.
            sync_interval_ms (int): Number of milliseconds between model synchronizations.
            warmup_steps (int): Number of steps to warm up by doing gradient allreduce before doing asynchronous
                model averaging. Use 0 to disable.
        """

        self.peer_selection_mode = peer_selection_mode
        self.sync_interval_ms = sync_interval_ms
        self.warmup_steps = warmup_steps

    def reify(self) -> AsyncModelAverageAlgorithm_Implementation:
        return AsyncModelAverageAlgorithm_Implementation(
            peer_selection_mode=self.peer_selection_mode,
            sync_interval_ms=self.sync_interval_ms,
            warmup_steps=self.warmup_steps,
        )
