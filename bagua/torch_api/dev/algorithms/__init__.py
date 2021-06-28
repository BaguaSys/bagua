#!/usr/bin/env python3

from bagua.torch_api.dev.distributed_dev import BaguaModule
from bagua.torch_api.dev.bucket import BaguaBucket
from bagua.torch_api.dev.tensor import BaguaTensor
from typing import List
import torch


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
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.

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
        parameters = bagua_module._bagua_build_params()
        tensor_groups = [[
           param.bagua_ensure_grad().to_bagua_tensor(name)
            for name, param in parameters.__reversed__()
        ]]
        # # TODO: consider optimizer groups
        # for name, param in reversed(list(bagua_module.named_parameters())):
        #     tensor = param.bagua_ensure_grad().to_bagua_tensor(name) # TODO: check duplicated names
        #     tensor_groups[0].append(tensor)
        return tensor_groups

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        """
        Given the bucketing suggestion from Bagua, return the actual Bagua buckets.
        The default implementation follows the suggestion to do the bucketing.

        Args:
            tensors: Bagua tensors grouped in different
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
        """Given a `BaguaModule`, return a hook function that will be executed on every
        parameter's gradient computation completion.

        Args:
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.

        Returns:
            A function that takes the name of a parameter.
        """
        def hook(name):

            ## bagua_module._bagua_tensor_map[name] does not have to be grad
            bagua_grad = bagua_module._bagua_tensor_map[name]
            bagua_grad.bagua_mark_communication_ready()
            # print(name, "in", bagua_grad._bagua_bucket.name, "ready")
        return hook

    def init_post_backward_hook(self, bagua_module: BaguaModule):
        """Given a `BaguaModule`, return a hook function that will be executed when the
        backward pass is done.

        Args:
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.

        Returns:
            A function that takes no argument.
        """
        def hook():
            bagua_module._bagua_backend.wait_pending_comm_ops()
        return hook

    def init_post_step_hook(self, bagua_module: BaguaModule):
        """Given a `BaguaModule`, return a hook function that will be executed when the
        ``optimizer.step()`` is done.

        Args:
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.

        Returns:
            A function that takes the optimizer that is called step().
        """
        def hook(optimizer: torch.optim.Optimizer):
            pass
        return hook

    def init_operations(
            self,
            bagua_module: BaguaModule,
            bucket: BaguaBucket,
    ):
        """Given a `BaguaModule`, and a Bagua bucket, register operations to be
        executed on the bucket.

        Args:
            bagua_module: A PyTorch module initialized by
                ``with_bagua(...)`` method.
            bucket: A single bucket to register operations.
        """
