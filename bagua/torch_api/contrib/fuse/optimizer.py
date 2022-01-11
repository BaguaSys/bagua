import torch
from typing import List, Dict, Optional, Any
import copy
import logging
import itertools
from bagua.torch_api.utils import check_contiguous, get_flattened_tensor
from collections import defaultdict
import gorilla


__all__ = ["fuse_optimizer", "fuse_step", "is_fused_optimizer"]


def flatten_tensors_with_closure(tensors, getter_closure, setter_closure):
    if len(tensors) == 0:
        return

    eff_tensors = [getter_closure(t) for t in tensors]
    if check_contiguous(eff_tensors):
        return

    flatten_tensor = get_flattened_tensor(eff_tensors)
    flatten_storage = flatten_tensor.storage()

    offset = 0
    for tensor in tensors:
        with torch.no_grad():
            z = torch.zeros_like(getter_closure(tensor))
            z.set_(flatten_storage, offset, z.shape)
            z._bagua_flattened_tensor = flatten_tensor
            setter_closure(tensor, z)

        offset += z.numel()
        logging.debug(f"flatten done {offset}, dtype: {z.dtype}")

    check_contiguous([getter_closure(t) for t in tensors])


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

            flatten_tensors_with_closure(
                params,
                getter_closure=lambda p: p.data,
                setter_closure=lambda p, new_data: setattr(p, "data", new_data),
            )

            flatten_tensors_with_closure(
                params,
                getter_closure=lambda p: p.grad.data,
                setter_closure=lambda p, new_grad: setattr(p.grad, "data", new_grad),
            )

            for name, tensors in state_tensors.items():
                flatten_tensors_with_closure(
                    params,
                    getter_closure=lambda p: optimizer.state[p][name].data,
                    setter_closure=lambda p, new_state: setattr(
                        optimizer.state[p][name], "data", new_state
                    ),
                )

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

    return grouped_indices


def _intersect(a: List[List[int]], b: List[List[int]]):
    c = [value for value in a if value in b]
    return c


def group_tensors(tensors: List[torch.Tensor], indices: List[int]):
    if len(indices) == 0:
        return

    to_group = [tensors[idx] for idx in indices]
    assert check_contiguous(to_group), "tensors grouped must be contiguous"

    total_size = sum([t.numel() for t in to_group])
    return to_group[0].storage(), total_size, to_group[0].dtype, to_group[0].device


def _create_tensor(storage, numel, dtype, device):
    logging.debug(f"Create new tensor with numel: {numel}, dtype: {dtype}")
    with torch.no_grad():
        tensor_view = torch.zeros(numel, dtype=dtype, device=device)
        tensor_view.set_(storage, 0, tensor_view.shape)
        return tensor_view


def _can_reuse_tensor(tensor, storage, numel, dtype, device):
    if tensor is None:
        return False

    return (
        tensor.storage().data_ptr() == storage.data_ptr()
        and tensor.numel() == numel
        and tensor.dtype == dtype
        and tensor.device == device
    )


def infer_state_tensors(optimizer, fused_param, original_params, name):
    fused_param_state = get_tensor_state(optimizer, fused_param, name)

    ungrouped = []
    offset = 0
    for p in original_params:
        if name in optimizer.state[p]:
            temp_tensor = get_tensor_state(optimizer, p, name)
        else:
            # use parameter meta to make a guess
            temp_tensor = p

        if temp_tensor.dtype != fused_param_state.dtype:
            logging.warning(
                "Fused optimizer failed to recover original parameter state, due to ambiguous parameter state datatype."
            )
            return

        z = fused_param_state.narrow(0, offset, temp_tensor.numel())
        z = z.view(temp_tensor.shape)
        offset += temp_tensor.numel()
        ungrouped.append(z)

    if offset != fused_param_state.numel():
        logging.warning(
            "Fused optimizer failed to recover original parameter state, due to ambiguous parameter state size."
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

    # Note: fused optimizer has its own copy of param groups and param state
    new_optimizer.param_groups = []
    for group in optimizer.param_groups:
        new_group = {"params": list(group["params"])}
        new_optimizer.add_param_group(new_group)

    new_optimizer.state = defaultdict(dict)

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

    for group, fused_group in zip(
        optimizer.param_groups, _fused_optimizer.param_groups
    ):
        sync_param_group_scalars(src_group=group, dst_group=fused_group)

        # Find params to fuse
        params = group["params"]
        weights = [p.data for p in params]
        grads = [p.grad for p in params]

        # Find original param state
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

        if len(grouped_indices) > 0:
            optimizer._bagua_fused_count += 1

        new_params = []
        fused_param_storages = {
            fp.storage().data_ptr(): fp
            for fp in fused_group["params"]
            if hasattr(fp, "_bagua_fused_param_ids")
        }

        active_param_ids = []
        for indices in grouped_indices:
            logging.debug(f"fuse params: {indices}")
            grouped_weight = group_tensors(weights, indices)
            grouped_grad = group_tensors(grads, indices)

            # create fused parameter
            with torch.no_grad():

                if _can_reuse_tensor(
                    fused_param_storages.get(grouped_weight[0].data_ptr()),
                    *grouped_weight,
                ):
                    fp = fused_param_storages[grouped_weight[0].data_ptr()]
                else:
                    fp = torch.nn.Parameter(
                        _create_tensor(*grouped_weight), requires_grad=False
                    )

                if not _can_reuse_tensor(fp.grad, *grouped_grad):
                    fp.grad = _create_tensor(*grouped_grad)

            fp._bagua_fused_param_ids = indices
            new_params.append(fp)
            active_param_ids.append(id(fp))

            # sync state tensors for fused optimizer
            for name, tensors in state_tensors.items():
                grouped_state = group_tensors(tensors, indices)

                if fp not in _fused_optimizer.state or not _can_reuse_tensor(
                    _fused_optimizer.state[fp].get(name, None), *grouped_state
                ):
                    _fused_optimizer.state[fp][name] = _create_tensor(*grouped_state)

            # sync state scalars for fused optimizer
            for name, scalar in state_scalars.items():
                _fused_optimizer.state[fp][name] = scalar

        # add non-contiguous params
        grouped_indices_flat = list(itertools.chain.from_iterable(grouped_indices))
        for idx, param in enumerate(params):
            if idx not in grouped_indices_flat:
                new_params.append(param)
                active_param_ids.append(id(param))

                for name, v in optimizer.state[param].items():
                    _fused_optimizer.state[param][name] = v

        # clear outdated states
        for fp in fused_group["params"]:
            if id(fp) not in active_param_ids and fp in _fused_optimizer.state:
                logging.debug("delete outdated params")
                del _fused_optimizer.state[fp]

        fused_group["params"] = new_params


def check_optimizer(optimizer):
    # make sure cloned attributes are not modified
    for attr in optimizer._bagua_cloned_attrs:
        if getattr(optimizer, attr) != getattr(optimizer._bagua_fused_optimizer, attr):
            logging.error(
                f"Should not change attribute {attr} in `optimizer.step(), maintain it in optimizer state.`"
            )


def sync_param_group_scalars(src_group, dst_group):
    for k, v in src_group.items():
        if k != "params":
            dst_group[k] = v

    for k, v in dst_group.items():
        if k not in src_group:
            dst_group[k] = None


def sync_optimizer_state(optimizer):
    # write back state for original params
    # Note: we should make sure every module parameter in original params groups has the right state
    _fused_optimizer = optimizer._bagua_fused_optimizer
    for group, fused_group in zip(
        optimizer.param_groups, _fused_optimizer.param_groups
    ):

        params = group["params"]
        fused_params = fused_group["params"]

        for fp in fused_params:
            if not hasattr(fp, "_bagua_fused_param_ids"):
                for name, v in _fused_optimizer.state[fp].items():
                    optimizer.state[fp][name] = v

            else:
                original_params = [params[i] for i in fp._bagua_fused_param_ids]

                for name, v in _fused_optimizer.state[fp].items():
                    if isinstance(v, torch.Tensor):
                        state_tensors = infer_state_tensors(
                            _fused_optimizer, fp, original_params, name
                        )

                        if state_tensors is not None:
                            for p, state in zip(original_params, state_tensors):
                                optimizer.state[p][name] = state
                    else:
                        for p in original_params:
                            optimizer.state[p][name] = v

                for p in original_params:
                    if len(optimizer.state[p]) != len(_fused_optimizer.state[fp]):
                        logging.warning("Something went wrong with optimizer state.")


def get_tensor_state(optimizer, param, name):
    s = optimizer.state[param][name]

    if not isinstance(s, torch.Tensor):
        logging.error(f"state {name} expected to be a tensor")

    return s


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
