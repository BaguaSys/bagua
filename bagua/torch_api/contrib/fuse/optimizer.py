import torch
from typing import List, Dict, Optional, Any
import copy
import logging
from functools import reduce
from bagua.torch_api.utils import check_contiguous, get_flattened_tensor
import gorilla


__all__ = ["fuse_optimizer", "fuse_step", "is_fused_optimizer"]


def flatten_params_and_states(optimizer: torch.optim.Optimizer):
    """
    Flatten parameter tensors in the same group into contiguous ones.
    """

    for group in optimizer.param_groups:
        type_params = {}
        # flatten by param group
        for param in group["params"]:

            params_of_type = type_params.get(param.type(), [])
            params_of_type.append(param)
            type_params[param.type()] = params_of_type

        for param_type, params in type_params.items():
            grads = [p.bagua_ensure_grad().grad for p in params]
            state_tensors, state_scalars = get_optimizer_param_states(optimizer, params)

            if state_tensors is None:
                continue

            flatten_tensors(params)
            flatten_tensors_with_closure(
                grads,
                params,
                getter_closure=lambda p: p.grad,
                setter_closure=lambda p, new_grad: setattr(p, "grad", new_grad),
            )

            for name, tensors in state_tensors.items():

                def set_state_fn(p, t):
                    optimizer.state[p][name] = t

                flatten_tensors_with_closure(
                    tensors,
                    params,
                    getter_closure=lambda p: optimizer.state[p][name],
                    setter_closure=set_state_fn,
                )
        torch.cuda.empty_cache()


def flatten_tensors(tensors: List[torch.Tensor]):
    """
    Flatten :attr:`tensors` into contiguous one.
    """
    if len(tensors) == 0:
        return

    if check_contiguous(tensors):
        return

    flatten_tensor = get_flattened_tensor(tensors)
    flatten_storage = flatten_tensor.storage()

    offset = 0
    for tensor in tensors:
        with torch.no_grad():
            tensor.set_(flatten_storage, offset, tensor.shape)

        offset += tensor.numel()
        logging.debug(f"flatten done {offset}")

    check_contiguous(tensors)


def flatten_tensors_with_closure(tensors, params, getter_closure, setter_closure):
    if len(tensors) == 0:
        return

    if check_contiguous(tensors):
        return

    flatten_tensor = get_flattened_tensor(tensors)
    flatten_storage = flatten_tensor.storage()

    offset = 0
    for tensor, param in zip(tensors, params):
        with torch.no_grad():
            z = torch.zeros_like(getter_closure(param))
            z.set_(flatten_storage, offset, z.shape)
            setter_closure(param, z)

        offset += tensor.numel()
        logging.debug(f"flatten with closure done {offset}")

    check_contiguous([getter_closure(p) for p in params])


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

    grouped_indices = []
    tmp_tensors = []
    tmp_indices = []

    for tensor, idx in sorted_tensor_list:
        if len(tmp_tensors) > 0 and not _is_contiguous_tensor(tensor, tmp_tensors[-1]):
            if len(tmp_tensors) > 1:
                grouped_indices.append(tmp_indices)
            tmp_tensors = []
            tmp_indices = []

        tmp_tensors.append(tensor)
        tmp_indices.append(idx)

    if len(tmp_tensors) > 1:
        grouped_indices.append(tmp_indices)

    return grouped_indices


def calculate_mutual_groups(tensors_list: List[List[torch.Tensor]]):
    constraints = []

    size = len(tensors_list[0])
    for tensors in tensors_list:
        assert size == len(
            tensors
        ), "Tensors to calculate mutual groups must have equal size."

        grouped_indices = _find_continuous_tensors(tensors)
        constraints.append(grouped_indices)

    if len(constraints) == 0:
        return constraints

    grouped_indices = constraints[0]
    for i in range(1, len(constraints)):
        grouped_indices = _intersect(grouped_indices, constraints[i])

    logging.debug(
        f"calculate mutual groups: {grouped_indices}, constraints: {constraints}"
    )
    return grouped_indices


def _intersect(a: List[List[int]], b: List[List[int]]):
    c = [value for value in a if value in b]
    return c


def group_tensors(tensors: List[torch.Tensor], indices: List[int]) -> torch.Tensor:
    if len(indices) == 0:
        return

    to_group = [tensors[idx] for idx in indices]
    assert check_contiguous(to_group), "tensors grouped must be contiguous"

    total_size = sum([t.numel() for t in to_group])
    with torch.no_grad():
        tensor_view = torch.zeros(
            total_size, dtype=to_group[0].dtype, device=to_group[0].device
        )
        tensor_view.set_(to_group[0].storage(), 0, tensor_view.shape)

        return tensor_view


def ungroup_tensor(
    tensor_view: torch.Tensor, tensors: List[torch.Tensor]
) -> Optional[List[torch.Tensor]]:
    """
    Ungroup :attr:`tensor_view` to a list of tensors that have same data types and sizes with :attr:`tensors`.
    """

    offset = 0
    ungrouped = []
    for tensor in tensors:
        if tensor_view.dtype != tensor.dtype:
            logging.warning(
                "Fused optimizer failed to recover parameter state from fused parameter state, due to mismatch between parameter datatype and parameter state datatype."
            )
            return

        z = torch.zeros_like(tensor)
        z.set_(tensor_view.storage(), offset, tensor.shape)

        offset += tensor.numel()
        ungrouped.append(z)

    if offset != tensor_view.numel():
        logging.warning(
            "Fused optimizer failed to recover parameter state from fused parameter state, due to mismatch between parameter size and parameter state size."
        )
        return

    return ungrouped


def fuse_optimizer(
    optimizer: torch.optim.Optimizer,
    do_flatten: bool = True,
    check_flatten: bool = True,
):
    """
    Convert any optimizer into a fused optimizer.

    A fused optimizer can fuse multiple parameter updates into one or a few updates. To achieve this, users need to:

    | 1) flatten multiple parameters in the same group into fused parameter by setting :attr:`do_flatten=True`,
         which is also the default behavior of a fused optimizer;
    | 2) perform a fused parameter update by calling :meth:`fuse_step`.

    This fused optimizer is implemented for general use. It can be used used in conjunction with
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

    optimizer._bagua_check_flatten = do_flatten and check_flatten
    optimizer._bagua_fused_count = 0
    optimizer._bagua_cloned_attrs = {}
    optimizer._bagua_fused_optimizer = make_optimizer_instance(optimizer)

    if do_flatten:
        flatten_params_and_states(optimizer)

    if not hasattr(optimizer, "fuse_step"):
        patch = gorilla.Patch(optimizer.__class__, "fuse_step", fuse_step)
        gorilla.apply(patch)

    return optimizer


def is_fused_optimizer(optimizer: torch.optim.Optimizer):
    """
    Checking if :attr:`optimizer` is a fused optimizer or not.
    """
    return hasattr(optimizer, "_bagua_fused_optimizer")


def make_optimizer_instance(optimizer: torch.optim.Optimizer):
    ignore_attrs = [
        "_bagua_check_flatten",
        "_bagua_fused_count",
        "_bagua_cloned_attrs",
    ]
    new_optimizer = copy.copy(optimizer)

    for attr in dir(optimizer):
        if attr not in ignore_attrs and attr not in dir(new_optimizer):
            logging.warning(
                f"Clone attribute {attr} to fused optimizer, should not modify it in `optimizer.step()`."
            )
            setattr(new_optimizer, attr, getattr(optimizer, attr))
            optimizer._bagua_cloned_attrs[attr] = getattr(optimizer, attr)

    # new_optimizer.param_groups = []
    # for group in optimizer.param_groups:
    #     new_group = {"params": list(group["params"])}
    #     new_optimizer.add_param_group(new_group)

    return new_optimizer


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
    check_optimizer(optimizer)
    sync_optimizer_state(optimizer)


def do_fuse(optimizer: torch.optim.Optimizer):
    _fused_optimizer = optimizer._bagua_fused_optimizer

    # Note: optimizer and fused optimizer share the same state, but different param groups
    _fused_optimizer.param_groups = []
    for index, group in enumerate(optimizer.param_groups):
        params = group["params"]

        weights = [p.data for p in params]
        grads = [p.grad for p in params]

        state_tensors, state_scalars = get_optimizer_param_states(optimizer, params)

        if state_tensors is None:
            continue

        check_flatten = optimizer._bagua_check_flatten
        if check_flatten and not check_contiguous(weights):
            logging.warning(
                "Parameter weights storage changed after flattened in fused optimizer, may degrade performance."
            )
            check_flatten = False

        if check_flatten and not check_contiguous(grads):
            logging.warning(
                "Parameter gradients storage changed after flattened in fused optimizer, may degrade performance."
            )
            check_flatten = False

        for name, tensors in state_tensors.items():
            if check_flatten and not check_contiguous(tensors):
                logging.warning(
                    "Parameter state {} storage changed after flattened in fused optimizer, may degrade performance.".format(
                        name
                    )
                )
                check_flatten = False

        grouped_indices = calculate_mutual_groups(
            [weights, grads] + list(state_tensors.values())
        )

        if len(grouped_indices) == 0:
            _fused_optimizer.add_param_group(group)
            continue

        optimizer._bagua_fused_count += 1

        new_params = []

        for indices in grouped_indices:
            grouped_weight = group_tensors(weights, indices)
            grouped_grad = group_tensors(grads, indices)

            grouped_states = {}
            for name, tensors in state_tensors.items():
                ts = group_tensors(tensors, indices)
                grouped_states[name] = ts

            with torch.no_grad():
                p = torch.nn.Parameter(grouped_weight, requires_grad=False)
                p.grad = grouped_grad
                p._bagua_fused_param_ids = indices

            # sync original param state to fused param state
            for name, ts in grouped_states.items():
                optimizer.state[p][name] = ts

            for name, v in state_scalars.items():
                optimizer.state[p][name] = v

            new_params.append(p)

        grouped_indices_flat = list(reduce(lambda x, y: x + y, grouped_indices))
        for idx, param in enumerate(params):
            if idx not in grouped_indices_flat:
                new_params.append(param)

        new_group = {"params": new_params}
        for k, v in group.items():
            if k != "params":
                new_group[k] = v

        _fused_optimizer.add_param_group(new_group)


def check_optimizer(optimizer):
    # make sure cloned attributes are not modified
    for attr in optimizer._bagua_cloned_attrs:
        if getattr(optimizer, attr) != getattr(optimizer._bagua_fused_optimizer, attr):
            logging.error(
                f"Should not change attribute {attr} in `optimizer.step(), maintain it in optimizer state.`"
            )


def sync_optimizer_state(optimizer):
    # write back state for original params
    # Note: we should make sure every module parameter in original params groups has the right state
    for group, fused_group in zip(
        optimizer.param_groups, optimizer._bagua_fused_optimizer.param_groups
    ):

        params = group["params"]
        fused_params = fused_group["params"]

        fused_state_tensors, fused_state_scalars = get_optimizer_param_states(
            optimizer, fused_params
        )

        for fp in fused_params:
            if not hasattr(fp, "_bagua_fused_param_ids"):
                continue

            original_params = [params[i] for i in fp._bagua_fused_param_ids]

            for name in fused_state_tensors.keys():
                state_tensors = ungroup_tensor(
                    optimizer.state[fp][name], original_params
                )

                if state_tensors is not None:
                    for p, state in zip(original_params, state_tensors):
                        optimizer.state[p][name] = state

            for name, v in fused_state_scalars.items():
                for p in original_params:
                    optimizer.state[p][name] = v

            # clear outdated state for fused param
            logging.debug("delete outdated params state")
            del optimizer.state[fp]


def get_optimizer_param_states(optimizer, params):
    state_tensors = {}  # type: Dict[str, List[torch.Tensor]]
    state_scalars = {}  # type: Dict[str, Any]

    state_tensor_names = set(
        [
            k
            for p in params
            for k, v in optimizer.state[p].items()
            if isinstance(v, torch.Tensor)
        ]
    )
    state_scalar_names = set(
        [
            k
            for p in params
            for k, v in optimizer.state[p].items()
            if not isinstance(v, torch.Tensor)
        ]
    )

    for name in state_tensor_names:
        tensors = []
        for p in params:
            if name not in optimizer.state[p]:
                logging.error(
                    f"Unexpected parameter state {name}, failed not fuse optimizer."
                )
                return None, None

            tensors.append(optimizer.state[p][name])

        state_tensors[name] = tensors

    for name in state_scalar_names:
        scalar = None

        for p in params:
            if name not in optimizer.state[p]:
                logging.error(
                    f"Unexpected parameter state {name}, failed not fuse optimizer."
                )
                return None, None

            if scalar is not None and scalar != optimizer.state[p][name]:
                logging.error(
                    f"Parameter state '{name}' does not match, failed not fuse optimizer."
                )
                return None, None

            state_scalars[name] = optimizer.state[p][name]

    return state_tensors, state_scalars
