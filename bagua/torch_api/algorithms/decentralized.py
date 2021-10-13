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
        `Decentralized SGD <https://bagua-tutorials.kwai-seattle.com/algorithms/decentralized>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            peer_selection_mode (str): Can be ``"all"`` or ``"shift_one"``. ``"all"`` means all workers'
                weights are averaged in each communication step. ``"shift_one"`` means each worker
                selects a different peer to do weights average in each communication step.
            communication_interval (int): Number of iterations between two communication steps.

        """
        self.hierarchical = hierarchical
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval

    def _should_communicate(self, bagua_module: BaguaModule) -> bool:
        cur_step = bagua_module.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        parameters = bagua_module.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name, bagua_module.bagua_module_name)
            for name, param in parameters.__reversed__()
        ]
        return self.tensors

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        all_tensors = []
        for idx, bucket in enumerate(tensors):
            all_tensors.extend(bucket)

        bagua_bucket = BaguaBucket(all_tensors, flatten=True, name=str(0))

        return [bagua_bucket]

    def init_forward_pre_hook(self, bagua_module: BaguaModule):
        def hook(input):
            if self._should_communicate(bagua_module):
                for tensor in self.tensors:
                    tensor.bagua_mark_communication_ready()

        return hook

    def init_backward_hook(self, bagua_module: BaguaModule):
        def hook(parameter_name, parameter):
            return

        return hook

    def init_post_backward_hook(self, bagua_module: BaguaModule):
        def hook():
            if self._should_communicate(bagua_module):
                bagua_module._bagua_backend.wait_pending_comm_ops()
                for bucket in bagua_module.bagua_buckets:
                    bucket.decentralized_synchronous_op_copy_back_peer_weight(
                        hierarchical=self.hierarchical, peer_weight=bucket._peer_weight
                    )

        return hook

    def _init_states(self, bucket: BaguaBucket):
        weight_tensor = bucket.flattened_tensor()
        bucket._peer_weight = weight_tensor.to_bagua_tensor("peer_weight")

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):
        self._init_states(bucket)
        torch.cuda.synchronize()
        bucket.clear_ops()
        bucket.append_decentralized_synchronous_op(
            peer_weight=bucket._peer_weight,
            hierarchical=self.hierarchical,
            peer_selection_mode=self.peer_selection_mode,
            group=bagua_module._bagua_process_group,
        )


class LowPrecisionDecentralizedAlgorithm(Algorithm):
    def __init__(self, hierarchical: bool = True, communication_interval: int = 1):
        """
        Create an instance of the
        `Low Precision Decentralized SGD <https://bagua-tutorials.kwai-seattle.com/algorithms/low-precision-decentralized>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
        """
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval

    def _should_communicate(self, bagua_module: BaguaModule) -> bool:
        cur_step = bagua_module.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

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
            if self._should_communicate(bagua_module):
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
        bucket.append_low_precision_decentralized_synchronous_op(
            weight=bucket._weight,
            left_peer_weight=bucket._left_peer_weight,
            right_peer_weight=bucket._right_peer_weight,
            hierarchical=self.hierarchical,
            compression="MinMaxUInt8",
            group=bagua_module._bagua_process_group,
        )
