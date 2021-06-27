#!/usr/bin/env python3

from bagua.torch_api.dev.bucket import BaguaBucket
from bagua.torch_api.dev.tensor import BaguaTensor
from bagua.torch_api.distributed_define import ReduceOp
from typing import List
import torch
from collections import OrderedDict

class Algorithm:
    def __init__(
        self,
    ):
        pass

    def need_reset(self) -> bool:
        "return True when we need to call init_buckets, init_hooks again. for example when we collect more info and want to rearrange the buckets"
        # TODO: previous buckets and hooks need to be cleared before reinit
        pass

    def init_tensors(self, bagua_module) -> List[List[BaguaTensor]]:
        """
        return an ordered dictionary of tensors to communicate
        every GPU should return in the same order
        """
        optimizers = bagua_module.bagua_optimizers
        tensor_groups = [[]]
        # TODO: consider optimizer groups
        named_param_list = bagua_module.named_parameters()
        for name, param in reversed(named_param_list):
            tensor = param.bagua_ensure_grad().to_bagua_tensor(name)
            tensor_groups[0].append(tensor)
        return tensor_groups

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        # TODO: real bucketing logic
        # TODO: use only specifies tensors, in first iteration, they are all separate buckets,
        # in the following iterations, the autotune server determines how to bucket them
        # the algorithm need to implement a tensors to buckets function
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(bucket, flatten=True, bucket_index=idx)
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_hooks(self, bagua_module) -> List:
        return []

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
