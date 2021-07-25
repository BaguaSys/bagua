#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm
import torch
from typing import List
from bagua.torch_api.tensor import BaguaTensor


class AsyncModelAverageAlgorithm(Algorithm):
    def __init__(self, peer_selection_mode: str = "all", sync_interval_ms: int = 500):
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

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        parameters = bagua_module.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name) for name, param in parameters.__reversed__()
        ]
        return self.tensors

    def init_forward_pre_hook(self, bagua_module: BaguaModule):
        def hook(input):
            if not hasattr(self, "async_handles"):
                self.async_handles = []
                for bucket in bagua_module.bagua_buckets:
                    #                    handle = bucket.execute_async(bucket.async_op)
                    handle = bucket._bagua_backend.schedule_comm_async(
                        bucket.backend_bucket, bucket.async_op
                    )
                    self.async_handles.append(handle)

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
        bucket.async_op = bucket.append_decentralized_asynchronous_op(
            peer_selection_mode=self.peer_selection_mode,
            sync_interval_ms=self.sync_interval_ms,
        )
