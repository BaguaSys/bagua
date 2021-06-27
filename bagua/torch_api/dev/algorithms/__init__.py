#!/usr/bin/env python3

from bagua.torch_api.dev.distributed_dev import BaguaModule
from bagua.torch_api.dev.bucket import BaguaBucket
from bagua.torch_api.dev.tensor import BaguaTensor
from bagua.torch_api.distributed_define import ReduceOp
from typing import List
import torch
from collections import OrderedDict

class Algorithm:
    """
    This is the base class that all Bagua algorithms inherit.

    It provides methods that can be override to implement different kinds of
    distributed algorithms.
    """

    def init_tensors(self, bagua_module: BaguaModule) -> List[List[BaguaTensor]]:
        """
        Given a BaguaModule, return Bagua tensors to be used in Bagua for later
        operations.

        Args:
            bagua_module (BaguaModule): A PyTorch module initialized by
                `with_bagua(...)` method.

        Returns:
            A list of list of Bagua tensors. The inner list represents a group
            of tensors (called tensor groups), that are preferred to be fused
            together.

        .. note::
            The tensor groups are useful when used with other optimizations such
            as `FusedOptimizer`, where the groups are the optimizers' parameter
            groups. Bagua will try to fuse the tensors in a way that both
            communication and fused optimization operations benefit as much as
            possible from the memory layout.
        """
        optimizers = bagua_module.bagua_optimizers
        tensor_groups = [[]]
        # TODO: @ganshaoduo consider optimizer groups
        for name, param in reversed(list(bagua_module.named_parameters())):
            tensor = param.bagua_ensure_grad().to_bagua_tensor(name) # TODO: check duplicated names
            tensor_groups[0].append(tensor)
        return tensor_groups

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        """
        Given the bucketing suggestion from Bagua, return the actual Bagua buckets.
        The default implementation follows the suggestion to do the bucketing.

        Args:
            tensors (List[List[BaguaTensor]]): Bagua tensors grouped in different
                lists, representing Bagua's suggestion on how to bucketing the
                tensors.

        Returns:
            A list of Bagua buckets.
        """
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(bucket, flatten=True, name=str(idx)) # TODO: check duplicated names
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_backward_hook(self, bagua_module: BaguaModule):
        """
        Given a `BaguaModule`,

        Return a function that takes the name of a parameter, and will be run when
        after the parameter's backward pass is done.
        """
        def hook(name):
            bagua_grad = bagua_module._bagua_tensor_map[name]
            bagua_grad.bagua_mark_communication_ready_on_current_stream()
        return hook

    def init_post_backward_hook(self, bagua_module):
        """Return a function that will be run when the whole model's backward pass is
        done.

        """
        def hook():
            bagua_module._bagua_backend.wait_pending_comm_ops()
        return hook

    def init_operations(
            self,
            bucket,
            inter_node_communicator,
            intra_node_communicator,
            global_communicator,
    ):
        pass


class DevelopAlgorithm(Algorithm):
    def __init__(self, hierarchical_reduce: bool, reduce_op: str = "avg"):
        self.reduce_op = reduce_op
        self.hierarchical_reduce = hierarchical_reduce

#     def init_tensors(self, bagua_module) -> List[BaguaTensor]:
#         tensors = []
#         for name, param in bagua_module.named_parameters(): # FIXME: we should keep track of communication ready order on hyperparamter server and bucket with that
#             tensors.append(param.to_bagua_tensor(name))
#         return tensors

#     def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
#         buckets = []
#         for tensor in tensors:
#             buckets.append(BaguaBucket([tensor]))
#         return buckets

    def init_operations(
            self,
            bucket,
            inter_node_communicator,
            intra_node_communicator,
            global_communicator,
    ):
        bucket.backend_bucket.append_centralized_synchronous_op(
            inter_node_communicator,
            intra_node_communicator,
            hierarchical=self.hierarchical_reduce,
            average=(self.reduce_op == ReduceOp.Average),
        )
