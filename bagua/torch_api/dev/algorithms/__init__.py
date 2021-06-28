#!/usr/bin/env python3

from bagua.torch_api.dev.distributed_dev import BaguaModule
from bagua.torch_api.dev.bucket import BaguaBucket
from bagua.torch_api.dev.tensor import BaguaTensor
from bagua.torch_api.distributed_define import ReduceOp
from typing import List
import torch
from collections import OrderedDict

def _build_params_for_reducer(bagua_model):
    # Build tuple of (module, parameter) for all parameters that require grads.
    modules_and_parameters = [
        [
            (module, parameter)
            for module_name, module in bagua_model.named_modules()
            for parameter in [
                    (f"{module_name}.{param_name}", param)
                    # Note that we access module.named_parameters instead of
                    # parameters(module). parameters(module) is only needed in the
                    # single-process multi device case, where it accesses replicated
                    # parameters through _former_parameters.
                for param_name, param in module.named_parameters(recurse=False)
                if param.requires_grad
                    and f"{module_name}.{param_name}" not in bagua_model.parameters_to_ignore
            ]
        ]
    ]

    # Deduplicate any parameters that might be shared across child modules.
    memo = set()
    modules_and_parameters = [
        # "p not in memo" is the deduplication check.
        # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
        [(m, p) for m, p in replica_mps if p not in memo and not memo.add(p)]
        for replica_mps in modules_and_parameters
    ]

    # Build list of parameters.
    parameters = [
        list(parameter for _, parameter in replica)
        for replica in modules_and_parameters
    ]

    # Checks if a module will produce a sparse gradient.
    def produces_sparse_gradient(module):
        if isinstance(module, torch.nn.Embedding) or isinstance(
                module, torch.nn.EmbeddingBag
        ):
            return module.sparse
        return False

    # Build list of booleans indicating whether or not to expect sparse
    # gradients for the corresponding parameters.
    expect_sparse_gradient = [
        list(produces_sparse_gradient(module) for module, _ in replica)
        for replica in modules_and_parameters
    ]

    return parameters, expect_sparse_gradient


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
        parameters, _ = _build_params_for_reducer(bagua_module)
        tensor_groups = [[
           param.bagua_ensure_grad().to_bagua_tensor(name) for inner_parameters in parameters for name, param in inner_parameters

        ]]
        # # TODO: @ganshaoduo consider optimizer groups
        # for name, param in reversed(list(bagua_module.named_parameters())):
        #     tensor = param.bagua_ensure_grad().to_bagua_tensor(name) # TODO: check duplicated names
        #     tensor_groups[0].append(tensor)
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
        """Given a `BaguaModule`, return a hook function that will be executed on every
        parameter's gradient computation completion.

        Args:
            bagua_module (BaguaModule): A PyTorch module initialized by
                `with_bagua(...)` method.

        Returns:
            A function that takes the name of a parameter.
        """
        def hook(name):
            bagua_grad = bagua_module._bagua_tensor_map[name]
            bagua_grad.bagua_mark_communication_ready()
        return hook

    def init_post_backward_hook(self, bagua_module: BaguaModule):
        """Given a `BaguaModule`, return a hook function that will be executed when the
        backward pass is done.

        Args:
            bagua_module (BaguaModule): A PyTorch module initialized by
                `with_bagua(...)` method.

        Returns:
            A function that takes no argument.
        """
        def hook():
            bagua_module._bagua_backend.wait_pending_comm_ops()
        return hook

    def init_operations(
            self,
            bagua_module: BaguaModule,
            bucket: BaguaBucket,
    ):
        """Given a `BaguaModule`, and a Bagua bucket, register operations to be
        executed on the bucket.

        Args:
            bagua_module (BaguaModule): A PyTorch module initialized by
                `with_bagua(...)` method.
            bucket (BaguaBucket): A single bucket to register operations.
        """
