#!/usr/bin/env python3

from bagua.torch_api.dev.bucket import BaguaBucket
from bagua.torch_api.dev.tensor import BaguaTensor
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

    def init_tensors(self, bagua_module) -> List[BaguaTensor]:
        """
        return an ordered dictionary of tensors to communicate
        every GPU should return in the same order
        """
        tensors = []
        for name, param in module.named_parameters():
            with torch.no_grad():
                t = torch.zeros_like(param.data)
                param.grad = t
            tensor = param.grad.to_bagua_tensor(name)
            tensors.append(tensor)
        return tensors

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        # TODO: real bucketing logic
        # TODO: use only specifies tensors, in first iteration, they are all separate buckets,
        # in the following iterations, the autotune server determines how to bucket them
        # the algorithm need to implement a tensors to buckets function
        # TODO:
        pass
        #     bucket = BaguaBucket([tensor])
        #     buckets.append(bucket)
        # return buckets
        buckets = []
        for bucket_dict in tensors:
            bucket = BaguaBucket(tensor_list)
            bucket.flatten_()
            buckets.append(bucket)
        return buckets

    def init_hooks(self, bagua_module) -> List:
        # register_backward_hook(
        #     {
        #         tensor.mark_ready()
        #     }
        # )
        pass

    def init_operations(
            self,
            bucket,
            inter_node_communicator,
            intra_node_communicator,
            global_communicator,
    ):
        # bucket.append(lambda bucket_name:
        #     calculate_m(bucket_name)
        # ).append_communication_op(
        #     ...
        # ).append_op(
        #     lambda bucket_name:
        #       ...
        # )
        pass


class DevelopAlgoritm(Algorithm):
    def __init__(self, hierarchical_reduce: bool, reduce_op: str = "avg"):
        self.reduce_op = reduce_op
        self.hierarchical_reduce = hierarchical_reduce

    def init_tensors(self, bagua_module) -> List[BaguaTensor]:
        tensors = []
        for name, param in bagua_module.named_parameters(): # FIXME: we should keep track of communication ready order on hyperparamter server and bucket with that
            tensor = param.to_bagua_tensor(name)
        return tensors

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        buckets = []
        for tensor in tensors:
            buckets.append(BaguaBucket([tensor]))
        return buckets
