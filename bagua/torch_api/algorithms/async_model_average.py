#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm
import torch
from typing import List
from bagua.torch_api.tensor import BaguaTensor


class AsyncModelAverageAlgorithm(Algorithm):
    def __init__(self, sync_interval_ms: int):
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
        self.sync_interval_ms = sync_interval_ms

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        parameters = bagua_module.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name) for name, param in parameters.__reversed__()
        ]
        return self.tensors

    def init_forward_pre_hook(self, bagua_module: BaguaModule):
        def hook(input):
            for tensor in self.tensors:
                tensor.bagua_mark_communication_ready()
        return hook

    def init_backward_hook(self, bagua_module: BaguaModule):
        def hook(parameter_name, parameter):
            return
        return hook

    def init_post_backward_hook(self, bagua_module: BaguaModule):
        def hook():
            bagua_module._bagua_backend.wait_pending_comm_ops()
            torch.cuda.synchronize()
            bagua_module._bagua_backend.execute_post_backward_comm_ops()
            bagua_module._bagua_backend.wait_pending_post_backward_comm_ops()

        return hook

    def init_operations(
            self,
            bagua_module: BaguaModule,
            bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        bucket.append_decentralized_synchronous_op(
            hierarchical=True,
            peer_selection_mode=self.peer_selection_mode,
            communication_interval=self.communication_interval,
        )
