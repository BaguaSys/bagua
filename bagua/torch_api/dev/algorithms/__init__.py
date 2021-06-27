#!/usr/bin/env python3

from bagua.torch_api.dev.bucket import BaguaBucket
from bagua.torch_api.dev.tensor import BaguaTensor
from typing import List
import torch

class Algorithm:
    def __init__(
        self,
    ):
        pass

    def need_reset(self) -> bool:
        "return True when we need to call init_buckets, init_hooks again. for example when we collect more info and want to rearrange the buckets"
        # TODO: previous buckets and hooks need to be cleared before reinit
        pass

    def init_buckets(self, module, optimizer) -> List:
        pass

    def init_hooks(self, module, optimizer) -> List:
        pass

    def init_operations(
        self,
        bucket,
        inter_node_communicator,
        intra_node_communicator,
        global_communicator,
    ):
        pass


def get_parameter_groups(optimizer: torch.optim.Optimizer):
    """
    Given a optimizer, return a dict containing Param => param_group_id
    """
    param_group_info = {}
    param_groups = [
        group for group in optimizer.param_groups
    ]
    for i, group in enumerate(param_groups):
        for param in group["params"]:
            param_group_info[param] = i
    return param_group_info


class DevelopAlgoritm(Algorithm):
    def __init__(self, hierarchical_reduce: bool, reduce_op: str = "avg"):
        self.reduce_op = reduce_op
        self.hierarchical_reduce = hierarchical_reduce


    def init_buckets(self, module, optimizer) -> List[BaguaBucket]:
        # TODO: real bucketing logic
        # TODO: use only specifies tensors, in first iteration, they are all separate buckets,
        # in the following iterations, the autotune server determines how to bucket them
        # the algorithm need to implement a tensors to buckets function
        buckets = []
        for param in module.parameters():
            tensor = param.to_bagua_tensor()
            bucket = BaguaBucket([tensor])
            buckets.append(bucket)
        return buckets
