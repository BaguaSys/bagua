import torch
from typing import List, Dict, Optional, Any
import copy
import logging
import itertools
from bagua.torch_api.utils import check_contiguous, get_flattened_tensor
from collections import defaultdict
import gorilla


__all__ = ["fuse_optimizer", "fuse_step", "is_fused_optimizer"]


def _pack_states(optimizer, params):
    assert len(params) > 0
    keys = set([k for p in params for k in optimizer.state[p].keys()])

    states = defaultdict(list)
    for k in keys:
        for param in params:
            assert k in optimizer.state[param]

            v = optimizer.state[param][k]
            if isinstance(v, torch.Tensor):
                states[k].append(v)
            else:
                if k not in states:
                    states[k] = v
                else:
                    assert v == states[k]
    return states


def _flatten_params_and_states_inplace(optimizer: torch.optim.Optimizer):
    """
    Flatten parameter tensors in the same group into contiguous ones.
    """
    params_groups = optimizer.param_groups

    for group in params_groups:
        param_dict = defaultdict(list)

        # For each param group:
        # 1. organize params by types
        for param in group["params"]:
            param_dict[param.type].append(param)

        # 2. flatten params with the same param type
        for param_type, params in param_dict.items():
            grads = [p.bagua_ensure_grad().grad for p in params]
            states = _pack_states(optimizer, params)

            flattened_params = get_flattened_tensor(params)
            flattened_grads = get_flattened_tensor(grads)

            flattened_states = {}
            for k in states.keys():
                if isinstance(states[k], list) and isinstance(
                    states[k][0], torch.Tensor
                ):
                    flattened_states[k] = get_flattened_tensor(states[k])

            offset = 0
            for p in params:
                with torch.no_grad():
                    p.set_(flattened_params.storage(), offset, p.shape)

                    z = torch.zeros_like(p)
                    z.set_(flattened_grads.storage(), offset, p.shape)
                    p.grad = z

                    for k in states.keys():
                        optimizer.state[p].set_(
                            flattened_states[k].storage(), offset, p.shape
                        )

                offset += p.numel()
                logging.debug(f"flatten done {offset}, dtype: {z.dtype}")

            check_contiguous(params)
            check_contiguous([p.bagua_ensure_grad().grad for p in params])
            for k in states.keys():
                if isinstance(states[k], list) and isinstance(
                    states[k][0], torch.Tensor
                ):
                    check_contiguous(states[k])

            torch.cuda.empty_cache()


def _is_contiguous_tensor(a: torch.Tensor, b: torch.Tensor):
    """
    Checking if tensor :attr:`a` and tensor :attr:`b` are contiguous.
    """
    size_a = a.numel() * a.element_size()
    size_b = b.numel() * b.element_size()

    return (a.data_ptr() == b.data_ptr() + size_b) or (
        b.data_ptr() == a.data_ptr() + size_a
    )


def _find_continuous_tensors(tensors: List[torch.Tensor]):
    tensor_list = zip(tensors, list(range(len(tensors))))
    sorted_tensor_list = sorted(tensor_list, key=lambda x: x[0].data_ptr())

    tensor_sizes = []
    grouped_indices = []
    tmp_tensors = []
    tmp_indices = []

    for tensor, idx in sorted_tensor_list:
        tensor_sizes.append(tensor.numel())
        if len(tmp_tensors) > 0 and not _is_contiguous_tensor(tensor, tmp_tensors[-1]):
            if len(tmp_tensors) > 1:
                grouped_indices.append(tmp_indices)
            tmp_tensors = []
            tmp_indices = []

        tmp_tensors.append(tensor)
        tmp_indices.append(idx)

    if len(tmp_tensors) > 1:
        grouped_indices.append(tmp_indices)

    return grouped_indices, tensor_sizes


def _find_mutual_continuous_tensors(weights, grads, states):
    cont_weight_indices, weight_sizes = _find_continuous_tensors(weights)
    cont_grad_indices, grad_sizes = _find_continuous_tensors(grads)

    if cont_weight_indices != cont_grad_indices or weight_sizes != grad_sizes:
        return []

    for k in states.keys():
        if isinstance(states[k], list) and isinstance(states[k][0], torch.Tensor):
            cont_state_indices, state_sizes = _find_continuous_tensors(states[k])

            if cont_state_indices != cont_weight_indices or weight_sizes != state_sizes:
                return []

    return cont_state_indices


def _group_tensors(tensors: List[torch.Tensor], indices: List[int]) -> torch.Tensor:
    if len(indices) == 0:
        return

    to_group = [tensors[idx] for idx in indices]
    assert check_contiguous(to_group), "tensors grouped must be contiguous"

    total_size = sum([t.numel() for t in to_group])
    # return to_group[0].storage(), total_size, to_group[0].dtype, to_group[0].device
    grouped_tensor = torch.Tensor(to_group[0].storage())
    assert grouped_tensor.dtype == to_group[0].dtype
    assert grouped_tensor.device == to_group[0].device
    assert grouped_tensor.numel() == total_size

    return grouped_tensor


def fuse_optimizer(optimizer: torch.optim.Optimizer):
    """
    Convert any optimizer into a fused optimizer.

    A fused optimizer can fuse multiple parameter updates into one or a few updates. To achieve this, users need to:

    | 1) flatten multiple parameters in the same group into fused parameter by setting :attr:`do_flatten=True`,
         which is also the default behavior of a fused optimizer;
    | 2) perform a fused parameter update by calling :meth:`fuse_step`.

    This fused optimizer is implemented for general use. It can be used in conjunction with
    a :class:`~bagua.torch_api.distributed.BaguaModule` as well as a
    `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel>`_
    wrapped module, or some other cases (not listed here).

    Args:
        optimizer (torch.optim.Optimizer): Any PyTorch optimizer.
        do_flatten (bool): Whether to flatten the parameters. The flatten operation will reset data pointers of
            parameter tensors so that they can be fused together. Default: ``True``.
        check_flatten (bool): When setting to ``True``, it enables fused optimizer to automatically check if
            parameter tensors are contiguous as they are flattened to. Can only work with :attr:`do_flatten=True`.
            Default: ``True``.

    Returns:
        A Fused optimizer.

    Example::
        >>> optimizer = torch.optim.Adadelta(model.parameters(), ....)
        >>> optimizer = bagua.torch_api.contrib.fuse_optimizer(optimizer, do_flatten=True)
        >>>
        >>> optimizer.fuse_step()

        When use in conjunction with a :class:`~bagua.torch_api.distributed.BaguaModule`, set :attr:`do_flatten=False`
        in :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua` explicitly:

        >>> optimizer = bagua.torch_api.contrib.fuse_optimizer(optimizer, do_flatten=True)
        >>> model = model.with_bagua([optimizer], GradientAllReduceAlgorithm(), do_flatten=False)
        >>>
        >>> optimizer.fuse_step()

    .. note::
        This function and :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua` method both will reset data
        pointers of module parameters by default. In order to perform a more effective fused parameter update,
        users need to disable bucket flattening in :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua`
        by setting its :attr:`do_flatten` to ``False``.

    .. note::
        A fuse optimizer does not change the original behaviors of :attr:`optimizer`, but enabling it to perform a
        fused parameter update through :meth:`fuse_step`. Users can still perform a normal parameter update through
        :meth:`step`.
    """

    if is_fused_optimizer(optimizer):
        raise RuntimeError("trying to fuse an optimizer twice!")

    optimizer._bagua_fused_optimizer = _create_fused_optimizer(optimizer)
    optimizer._bagua_fused_count = 0

    if not hasattr(optimizer, "fuse_step"):
        patch = gorilla.Patch(optimizer.__class__, "fuse_step", fuse_step)
        gorilla.apply(patch)

    return optimizer


def is_fused_optimizer(optimizer: torch.optim.Optimizer):
    """
    Checking if :attr:`optimizer` is a fused optimizer or not.
    """
    return hasattr(optimizer, "_bagua_fused_optimizer")


def _create_fused_optimizer(optimizer: torch.optim.Optimizer):
    if optimizer.defaults.get("fused"):
        return optimizer

    bagua_optimizer = copy.copy(optimizer)
    _flatten_params_and_states_inplace(bagua_optimizer)
    return bagua_optimizer


def fuse_step(optimizer: torch.optim.Optimizer, closure=None):
    r"""Perform a fused parameter update.

    This operation will fuse multiple contiguous parameters into a fused parameter, by creating a tensor
    view sharing the same underlying storage with them, and then perform parameter update on fused parameters.
    If none of the parameter tensors are contiguous, this operation is equivalent to :meth:`step`.

    Args:
        optimizer: A fused optimizer.
        closure (Callable): A closure that reevaluates the model and
            returns the loss. Optional for most optimizers.

    .. note::
        This function will not modify the storage of parameter tensors.
    """

    assert is_fused_optimizer(
        optimizer
    ), "Should init fused optimizer by calling `fuse_optimizer`."

    do_fuse(optimizer)
    optimizer._bagua_fused_optimizer.step(closure)
    sync_optimizer_state(optimizer)


def do_fuse(optimizer: torch.optim.Optimizer):

    fused_param_groups = []
    fused_optimizer_state = {}

    # create fused param groups and states in the fly
    for group in optimizer.param_groups:
        new_group = defaultdict(list)
        for key, value in group.items():
            if key == "params":
                weights = [p.data for p in group["params"]]
                grads = [p.grad for p in group["params"]]
                states = _pack_states(optimizer, group["params"])

                cont_indices = _find_mutual_continuous_tensors(weights, grads, states)
                if len(cont_indices) == 0:
                    continue

                optimizer._bagua_fused_count += 1
                for indices in grouped_indices:
                    logging.debug(f"fuse params: {indices}")
                    grouped_weight = group_tensors(weights, indices)
                    grouped_grad = group_tensors(grads, indices)

                    fused_param = torch.nn.Parameter(
                        grouped_weight, requires_grad=False
                    )
                    fused_param.grad = grouped_grad
                    new_group["params"].append(fused_param)

                    for k in states.keys():
                        if isinstance(states[k], list) and isinstance(
                            states[k][0], torch.Tensor
                        ):
                            grouped_state = group_tensors(states[k], indices)
                            fused_optimizer_state[fused_param][k] = grouped_state
                        else:
                            fused_optimizer_state[fused_param][k] = states[k]
            else:
                new_group[key] = value

            fused_param_groups.append(new_group)

        optimizer._bagua_fused_optimizer.param_groups = fused_param_groups
        optimizer._bagua_fused_optimizer.state = fused_optimizer_state
