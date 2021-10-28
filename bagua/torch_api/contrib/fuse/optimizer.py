import torch
from typing import List
import copy
import logging
from functools import reduce
from bagua.torch_api.utils import check_contiguous, get_flattened_tensor
import gorilla


__all__ = ["fuse_optimizer", "fuse_step"]


def flatten_param_and_states(optimizer: torch.optim.Optimizer):
    # _supported_params_types = [
    #     "torch.cuda.FloatTensor",
    #     "torch.cuda.HalfTensor",
    #     "torch.FloatTensor",
    #     "torch.HalfTensor",
    # ]

    type_params = {}
    for group in optimizer.param_groups:
        for param in group["params"]:

            params_of_type = type_params.get(param.type(), [])
            params_of_type.append(param)
            type_params[param.type()] = params_of_type

    for type, params in type_params.items():
        grads = [p.bagua_ensure_grad().grad for p in params]

        state_tensors, state_scalars, succ = _get_state_by_name(optimizer.state, params)

        if not succ:
            continue

        _flatten_(params)
        _flatten_(grads)
        logging.debug(f"flatten {type} params done")

        for name, tensors in state_tensors.items():
            _flatten_(tensors)
            logging.debug(f"flatten state {name} done")

        grads = [p.grad for p in params]
        state_tensors, state_scalars, succ = _get_state_by_name(optimizer.state, params)

        check_contiguous(params)
        check_contiguous(grads)
        for name, tensors in state_tensors.items():
            check_contiguous(tensors)


def _flatten_(tensors: List[torch.Tensor]):
    if len(tensors) == 0:
        return

    if not check_contiguous(tensors):
        flatten_tensor = get_flattened_tensor(tensors)
        flatten_storage = flatten_tensor.storage()

        offset = 0
        for tensor in tensors:
            tensor.set_(flatten_storage, offset, tensor.shape)
            offset += tensor.numel()
            logging.debug(f"flatten done {offset}")

    # flatten effective tensors if exist
    has_effective_tensor = all([tensor.is_bagua_tensor() for tensor in tensors])
    if has_effective_tensor:
        eff_tensors = [tensor.bagua_getter_closure() for tensor in tensors]

        if not check_contiguous(eff_tensors):
            flatten_eff_tensor = get_flattened_tensor(eff_tensors)
            flatten_eff_storage = flatten_eff_tensor.storage()

            offset = 0
            for tensor in tensors:
                tensor.bagua_set_storage(flatten_eff_storage, offset)
                offset += tensor.bagua_getter_closure().numel()
                logging.debug(f"flatten effective tensor done {offset}")


def is_contiguous_tensor(a: torch.Tensor, b: torch.Tensor):
    size_a = a.numel() * a.element_size()
    size_b = b.numel() * b.element_size()

    return (a.data_ptr() == b.data_ptr() + size_a) or (
        b.data_ptr() == a.data_ptr() + size_b
    )


def group_continuous_tensors(tensors: List[torch.Tensor]):
    tensor_list = zip(tensors, list(range(len(tensors))))
    sorted_tensor_list = sorted(tensor_list, key=lambda x: x[0].data_ptr())

    grouped_indices = []
    tmp_tensors = []
    tmp_indices = []

    for tensor, idx in sorted_tensor_list:
        if len(tmp_tensors) > 0 and not is_contiguous_tensor(tensor, tmp_tensors[-1]):
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

        grouped_indices = group_continuous_tensors(tensors)

        constraints.append(grouped_indices)

    # no constraints, group them all
    if len(constraints) == 0:
        return constraints

    grouped_indices = constraints[0]
    for i in range(1, len(constraints)):
        grouped_indices = intersect(grouped_indices, constraints[i])

    logging.debug(
        f"calculate mutual groups: {grouped_indices}, constraints: {constraints}"
    )
    return grouped_indices


def intersect(a: List[List[int]], b: List[List[int]]):
    c = [value for value in a if value in b]
    return c


def group_tensors(tensors: List[torch.Tensor], grouped_indices: List[List[int]]):
    logging.debug(f"Ready to group tensors, grouped indices: {grouped_indices}.")

    grouped_tensors = []
    for indices in grouped_indices:
        ts = [tensors[idx] for idx in indices]
        assert check_contiguous(ts), "tensors grouped must be contiguous"

        total_size = sum([t.numel() for t in ts])
        with torch.no_grad():
            tensor_view = torch.zeros(total_size, dtype=ts[0].dtype).to(ts[0].device)
            tensor_view.set_(ts[0].storage(), 0, tensor_view.shape)

            grouped_tensors.append(tensor_view)

    return grouped_tensors


def fuse_optimizer(
    optimizer: torch.optim.Optimizer,
    do_flatten: bool = True,
    check_flatten: bool = True,
):
    """
    Convert any optimizer into a fused optimizer.

    A fused optimizer can fuse multiple parameter updates into one. To achieve this, users need to:

    | 1) flatten parameter tensors in the same group into contiguous ones by setting :attr:`do_flatten=True`,
         which is also the default behavior of a fused optimizer;
    | 2) fuse multiple parameter tensors contiguous in memory into one and perform fused parameter
         updates by calling :meth:`fuse_step`.

    This fused optimizer is implemented for general use. It can be used used in conjunction with
    a :class:`~bagua.torch_api.distributed.BaguaModule` as well as a
    `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel>`_
    wrapped module, or some other cases (not listed here).

    Args:
        optimizer (torch.optim.Optimizer): Any PyTorch optimizer.
        do_flatten (bool): Whether to flatten the parameters. The flatten operation will reset data pointers of
            parameter tensors so that they could perform fused updates.  Default: ``True``.
        check_flatten (bool): When setting to ``True``, it enables fused optimizer to automatically check if
            parameter tensors are flattened as they are required to, i.e. by setting :attr:`do_flatten=True`.
            You can disable this check under production. Default: ``True``.

    Returns:
        Fused optimizer.

    Example::
        To use in conjunction with a :class:`~bagua.torch_api.distributed.BaguaModule`:

        >>> optimizer = torch.optim.Adadelta(model.parameters(), ....)
        >>> optimizer = bagua.torch_api.contrib.fuse_optimizer(optimizer)
        >>> model = model.with_bagua([optimizer], GradientAllReduceAlgorithm(), do_flatten=False)
        >>>
        >>> optimizer.fuse_step()
        >>>

        Otherwise:

        >>> optimizer = torch.optim.Adadelta(model.parameters(), ....)
        >>> optimizer = bagua.torch_api.contrib.fuse_optimizer(optimizer)
        >>>
        >>> optimizer.fuse_step()
        >>>

    .. note::
        This function and :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua` method both will reset data
        pointers of module parameters by default. In order to perform fused parameter updates, you need to set
        set :attr:`do_flatten=False` in :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua` when initializing
        the :class:`~bagua.torch_api.distributed.BaguaModule`.

    .. note::
        A fuse optimizer will not change the original behaviors of its :attr:`optimizer`, but enabling it to perform
        fused parameter updates by calling :meth:`fuse_step`.
    """

    fused_optimizer = copy.copy(optimizer)

    optimizer._bagua_fused_optimizer = fused_optimizer
    optimizer._bagua_do_flatten = do_flatten
    optimizer._bagua_check_flatten = do_flatten and check_flatten

    if do_flatten:
        flatten_param_and_states(optimizer)

    if not hasattr(optimizer, "fuse_step"):
        patch = gorilla.Patch(optimizer.__class__, "fuse_step", fuse_step)
        gorilla.apply(patch)

    return optimizer


def is_fused_optimizer(optimizer: torch.optim.Optimizer):
    return hasattr(optimizer, "_bagua_fused_optimizer")


def fuse_step(optimizer: torch.optim.Optimizer, closure=None):
    r"""Fuse parameters and perform a single optimization step (parameter update).

    This operation will create a tensor view for multiple contiguous parameter tensors and perform
    parameter update using this view. If none of the parameter tensors are contiguous, this operation
    is equivalent to :meth:`step`.

    Args:
        optimizer: A fused optimizer.
        closure (Callable): A closure that reevaluates the model and
            returns the loss. Optional for most optimizers.

    .. note::
        This function will not modify metadata of parameter tensors.
    """

    if is_fused_optimizer(optimizer):
        do_fuse(
            optimizer._bagua_fused_optimizer,
            check_flatten=optimizer._bagua_check_flatten,
        )
        return optimizer._bagua_fused_optimizer.step(closure)

    logging.debug("Fall back to normal step")
    return optimizer.step(closure)


def do_fuse(optimizer: torch.optim.Optimizer, check_flatten):
    for index, group in enumerate(optimizer.param_groups):
        params = group["params"]

        weights = [p.data for p in params]
        grads = [p.grad for p in params]

        state_tensors, state_scalars, succ = _get_state_by_name(optimizer.state, params)

        if not succ:
            continue

        if check_flatten and not check_contiguous(weights):
            logging.warn(
                "Parameter weights are not contiguous in memory, should not change the data pointers elsewhere."
            )
            check_flatten = False

        if check_flatten and not check_contiguous(grads):
            logging.warn(
                "Parameter weights are not contiguous in memory, should not change the data pointers elsewhere."
            )
            check_flatten = False

        for name, tensors in state_tensors.items():
            if check_flatten and not check_contiguous(tensors):
                logging.warn(
                    "Parameter state {} are not contiguous in memory, should not change the data pointers elsewhere.".format(
                        name
                    )
                )
                check_flatten = False

        grouped_indices = calculate_mutual_groups(
            [weights, grads] + list(state_tensors.values())
        )

        if len(grouped_indices) > 0:
            # group params
            grouped_weights = group_tensors(weights, grouped_indices)
            grouped_grads = group_tensors(grads, grouped_indices)

            grouped_states = {}
            for name, tensors in state_tensors.items():
                ts = group_tensors(tensors, grouped_indices)
                grouped_states[name] = ts

            new_params = []
            for i in range(len(grouped_weights)):
                with torch.no_grad():
                    p = torch.nn.Parameter(grouped_weights[i], requires_grad=False)
                    p.grad = grouped_grads[i]

                new_params.append(p)

                for name, ts in grouped_states.items():
                    optimizer.state[p][name] = ts[i]

                for name, v in state_scalars.items():
                    optimizer.state[p][name] = v

            # add other params and remove dup states
            grouped_indices_flat = list(reduce(lambda x, y: x + y, grouped_indices))
            for idx, param in enumerate(params):
                if idx not in grouped_indices_flat:
                    new_params.append(param)
                elif optimizer.state.get(param) is not None:
                    logging.debug(f"Ready to delete state for param {idx}")
                    del optimizer.state[param]

            group["params"] = new_params


def _get_state_by_name(optimizer_state, params):
    state_tensors = {}
    state_scalars = {}

    if len(optimizer_state) > 0:
        state_tensors = {
            name: []
            for name, value in optimizer_state[params[0]].items()
            if isinstance(value, torch.Tensor)
        }
        state_scalars = {
            name: value
            for name, value in optimizer_state[params[0]].items()
            if not isinstance(value, torch.Tensor)
        }

        for p in params:
            st = optimizer_state[p]

            for name, value in st.items():
                if isinstance(value, torch.Tensor):
                    if state_tensors.get(name) is None:
                        logging.error(
                            f"Unexpected tensor in state {name}, could not fuse optimizer."
                        )
                        return None, None, False

                    state_tensors[name].append(value)
                else:
                    if state_scalars.get(name) is None:
                        logging.error(
                            f"Unexpected scalar value in state {name}, could not fuse optimizer."
                        )
                        return None, None, False

                    if value != state_scalars[name]:
                        logging.error(
                            f"Parameter state '{name}' does not match, could not fuse optimizer."
                        )
                        return None, None, False

        for name, tensors in state_tensors.items():
            if len(tensors) != len(params):
                logging.error(
                    f"Parameter state '{name}' does not match, could not fuse optimizer."
                )
                return None, None, False

    return state_tensors, state_scalars, True
