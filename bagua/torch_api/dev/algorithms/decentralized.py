#!/usr/bin/env python3

from bagua.torch_api.dev.bucket import BaguaBucket
from bagua.torch_api.dev.tensor import BaguaTensor
from bagua.torch_api.dev.distributed_dev import BaguaModule
from bagua.torch_api.dev.algorithms import Algorithm
from typing import List
import torch


class DecentralizedAllReduceAlgorithm(Algorithm):
    def __init__(
        self,
        peer_selection_mode: str = "all",
        compression: str = None,
        communication_interval: int = 1,
    ):
        """
        Create an instance of the
        `Decentralized <https://baguasys.github.io/tutorials/algorithms/decentralized.html>`_
        algorithm.

        Args:
            peer_selection_mode (str): xxx.
        """
        self.peer_selection_mode = peer_selection_mode
        self.compression = compression
        self.communication_interval = communication_interval

    def init_tensors(self, bagua_module: BaguaModule) -> List[List[BaguaTensor]]:

        optimizers = bagua_module.bagua_optimizers
        parameters = bagua_module._bagua_build_params()
        tensor_groups = [
            [param.to_bagua_tensor(name) for name, param in parameters.__reversed__()]
        ]
        # # TODO: consider optimizer groups
        # for name, param in reversed(list(bagua_module.named_parameters())):
        #     tensor = param.bagua_ensure_grad().to_bagua_tensor(name) # TODO: check duplicated names
        #     tensor_groups[0].append(tensor)
        return tensor_groups

    def init_backward_hook(self, bagua_module: BaguaModule):
        def hook(name):
            bagua_weight = bagua_module._bagua_tensor_map[name]
            bagua_weight.bagua_mark_communication_ready_eager()

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
        bucket.backend_bucket.clear_ops()
        bucket.backend_bucket.append_decentralized_synchronous_op(
            bagua_module.bagua_inter_node_communicator,
            bagua_module.bagua_intra_node_communicator,
            hierarchical=True,
            compression=self.compression,
            peer_selection_mode=self.peer_selection_mode,
            communication_interval=self.communication_interval,
        )
