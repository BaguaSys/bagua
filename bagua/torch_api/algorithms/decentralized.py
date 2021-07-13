#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm
from typing import List
import torch


class DecentralizedAlgorithm(Algorithm):
    def __init__(
        self,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
        communication_interval: int = 1,
    ):
        """
        Create an instance of the
        `Decentralized <https://baguasys.github.io/tutorials/algorithms/decentralized.html>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            peer_selection_mode (str): Can be "all" or "shift_one". "all" means all workers'
                weights are averaged in each communication step. "shift_one" means each worker
                selects a different peer to do weights average in each communication step.
            communication_interval (int): Number of iterations between two communication steps.
        """
        self.hierarchical = hierarchical
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        parameters = bagua_module.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name, bagua_module.bagua_module_name)
            for name, param in parameters.__reversed__()
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
            hierarchical=self.hierarchical,
            peer_selection_mode=self.peer_selection_mode,
            communication_interval=self.communication_interval,
        )


class LowPrecisionDecentralizedAlgorithm(Algorithm):
    def __init__(self, hierarchical: bool = True, communication_interval: int = 1):
        """
        Create an instance of the
        `Difference Compression Decentralized <https://arxiv.org/pdf/1803.06443.pdf>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
        """
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        parameters = bagua_module.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name, bagua_module.bagua_module_name)
            for name, param in parameters.__reversed__()
        ]
        optimizer_param_ids = [
            id(param)
            for optimizer in bagua_module.bagua_optimizers
            for group in optimizer.param_groups
            for param in group["params"]
        ]

        for name, param in parameters:
            if id(param) not in optimizer_param_ids:
                raise RuntimeError(
                    f"Module parameter {name} is not used by your optimizer(s), need to exclude it "
                    "by adding the parameter name to the `List` attribute `_bagua_params_and_buffers_to_ignore` "
                    "of your module."
                )
        return self.tensors

    def init_backward_hook(self, bagua_module: BaguaModule):
        def hook(parameter_name, parameter):
            pass

        return hook

    def init_post_backward_hook(self, bagua_module: BaguaModule):
        def hook():
            pass

        return hook

    def init_post_optimizer_step_hook(self, bagua_module: BaguaModule):
        def hook(optimizer: torch.optim.Optimizer):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.is_bagua_tensor():
                        param.bagua_mark_communication_ready()

            bagua_module._bagua_backend.wait_pending_comm_ops()

        return hook

    def _init_states(self, bucket: BaguaBucket):
        weight_tensor = bucket.flattened_tensor()
        left_peer_weight_tensor = bucket.flattened_tensor()
        right_peer_weight_tensor = bucket.flattened_tensor()

        bucket._weight = weight_tensor.to_bagua_tensor("weight")
        bucket._left_peer_weight = left_peer_weight_tensor.to_bagua_tensor(
            "left_peer_weight"
        )
        bucket._right_peer_weight = right_peer_weight_tensor.to_bagua_tensor(
            "right_peer_weight"
        )

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):
        self._init_states(bucket)
        torch.cuda.synchronize()
        bucket.clear_ops()
        bucket.append_decentralized_synchronous_op(
            hierarchical=self.hierarchical,
            peer_selection_mode="ring",
            communication_interval=self.communication_interval,
            compression="MinMaxUInt8",
            weight=bucket._weight,
            left_peer_weight=bucket._left_peer_weight,
            right_peer_weight=bucket._right_peer_weight,
        )
