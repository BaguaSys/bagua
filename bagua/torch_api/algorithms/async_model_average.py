#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm
import torch
from typing import List
from bagua.torch_api.tensor import BaguaTensor
import threading
import time


class AsyncModelAverageAlgorithm(Algorithm):
    def __init__(
        self,
        peer_selection_mode: str = "all",
        sync_interval_ms: int = 500,
        stop_grace_period_secs: int = 5,
    ):
        """
        Create an instance of the
        `AsyncModelAverage <https://baguasys.github.io/tutorials/algorithms/async-model-average.html>`_
        algorithm.

        The currently implementation is experimental, and has some restrictions on the training scenarios.
        Since with an async algorithm, each worker can be in different iterations, the current implementation
        assumes the data are in a endless stream, and there is no concept of an "epoch".

        Args:
            sync_interval_ms (int): How many milliseconds between two model synchronizations.
        """
        self.peer_selection_mode = peer_selection_mode
        self.sync_interval_ms = sync_interval_ms
        self.stop_grace_period_secs = stop_grace_period_secs
        self.stop_event = threading.Event()

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        parameters = bagua_module.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name, bagua_module.bagua_module_name)
            for name, param in parameters.__reversed__()
        ]
        return self.tensors

    def init_forward_pre_hook(self, bagua_module: BaguaModule):
        def hook(input):
            if not hasattr(self, "worker"):
                self.worker = threading.Thread(
                    target=self.run_async_loop, args=[bagua_module]
                )
                self.worker.start()

        return hook

    def init_backward_hook(self, bagua_module: BaguaModule):
        def hook(parameter_name, parameter):
            pass

        return hook

    def init_post_backward_hook(self, bagua_module: BaguaModule):
        def hook():
            pass

        return hook

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        bucket.append_decentralized_asynchronous_op(
            peer_selection_mode=self.peer_selection_mode,
        )

    def barrier(self, bagua_module: BaguaModule):
        self.stop_event.set()
        time.sleep(self.stop_grace_period_secs)
        bagua_module._bagua_backend.global_communicator.abort()
        self.worker.join()

    def run_async_loop(self, bagua_module: BaguaModule):
        step = 0
        while not self.stop_event.is_set():
            for bucket in bagua_module.bagua_buckets:
                for tensor in bucket.tensors:
                    tensor.bagua_mark_communication_ready()

            bagua_module._bagua_backend.wait_pending_comm_ops()

            time.sleep(self.sync_interval_ms / 1000)
            step += 1
