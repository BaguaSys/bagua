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
            hierarchical=self.hierarchical,
            peer_selection_mode=self.peer_selection_mode,
            communication_interval=self.communication_interval,
        )


class LowPrecisionDecentralizedAlgorithm(Algorithm):
    def __init__(self, hierarchical: bool = True):
        """
        Create an instance of the
        `Difference Compression Decentralized <https://arxiv.org/pdf/1803.06443.pdf>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
        """
        self.hierarchical = hierarchical

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        parameters = bagua_module.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name) for name, param in parameters.__reversed__()
        ]
        return self.tensors

    def init_backward_hook(self, bagua_module: BaguaModule):
        def hook(parameter_name, parameter):
            pass

        return hook

    def init_post_backward_hook(self, bagua_module: BaguaModule):
        def hook():
            for bucket in bagua_module.bagua_buckets:
                for tensor in bucket.tensors:
                    tensor.bagua_mark_communication_ready()

            bagua_module._bagua_backend.wait_pending_comm_ops()

        return hook

    def init_post_optimizer_step_hook(self, bagua_module: BaguaModule):
        def hook(optimizer: torch.optim.Optimizer):
            bagua_module._bagua_backend.execute_post_optimizer_step_comm_ops()
            bagua_module._bagua_backend.wait_pending_post_optimizer_step_comm_ops()

        return hook

    def _init_states(self, bucket: BaguaBucket):
        bucket_flattened_tensor = bucket.flattened_tensor()

        left_peer_weight_tensor = bucket_flattened_tensor.detach().clone()
        right_peer_weight_tensor = bucket_flattened_tensor.detach().clone()

        bucket.set_state("left_peer_weight", left_peer_weight_tensor)
        bucket.set_state("right_peer_weight", right_peer_weight_tensor)

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
            compression="MinMaxUInt8",
        )
