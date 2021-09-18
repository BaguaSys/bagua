import torch
from typing import List
import copy
import logging
from functools import reduce
from bagua.torch_api.utils import check_contiguous, get_flattened_tensor
import gorilla


def flatten_param_and_states(optimizer: torch.optim.Optimizer):
    _supported_params_types = [
        "torch.cuda.FloatTensor",
        "torch.cuda.HalfTensor",
        "torch.FloatTensor",
        "torch.HalfTensor",
    ]

    for group in optimizer.param_groups:

        for param_type in _supported_params_types:
            params = [param for param in group["params"] if param.type() == param_type]

            weights = [p.data for p in params]
            grads = [p.bagua_ensure_grad() for p in params]

            state_tensors, state_scalars, succ = _get_state_by_name(
                optimizer.state, params
            )

            if not succ:
                continue

            grouped_indices = calculate_mutual_groups(
                [weights, grads] + list(state_tensors.values()), op=union
            )

            print(f"param type: {param_type}, grouped indices: {grouped_indices}")
            if len(grouped_indices) > 0:
                if not check_duplicated(grouped_indices):
                    flatten_tensors(params, grouped_indices)
                    flatten_tensors(grads, grouped_indices)

                    for name, tensors in state_tensors.items():
                        flatten_tensors(tensors, grouped_indices)


def flatten_module(module: torch.nn.Module):
    print("Ready to flatten module")
    weights = [p for p in module.parameters()]
    grads = [p.bagua_ensure_grad() for p in module.parameters()]

    grouped_indices = calculate_mutual_groups([weights, grads], op=union)
    if len(grouped_indices) > 0:
        if not check_duplicated(grouped_indices):
            flatten_tensors(weights, grouped_indices)
            flatten_tensors(grads, grouped_indices)


def flatten_tensors(tensors, grouped_indices: List[List[int]]):
    tensors_grouped_indices = group_continuous_tensors(tensors)

    grouped_indices = difference(grouped_indices, tensors_grouped_indices)
    print(
        f"ready to flatten tensors, result grouped indices: {grouped_indices}, tensors grouped indices: {tensors_grouped_indices}"
    )
    for indices in grouped_indices:
        to_flatten = [tensors[i] for i in indices]
        flattened_tensor = get_flattened_tensor(to_flatten)
        flattened_storage = flattened_tensor.storage()

        print(f"before flatten: {to_flatten}, after flatten: {flattened_tensor}")
        offset = 0
        with torch.no_grad():
            for tensor in to_flatten:
                tensor.set_(flattened_storage, offset, tensor.shape)
                offset += tensor.numel()
                print(f"flatten param done {offset}")

        assert check_contiguous(to_flatten)


def is_contiguous_tensor(a: torch.Tensor, b: torch.Tensor):
    allocate_size_a = (
        a.bagua_tensor.num_elem_allocated() if hasattr(a, "bagua_tensor") else a.numel()
    )
    allocate_size_b = (
        b.bagua_tensor.num_elem_allocated() if hasattr(b, "bagua_tensor") else b.numel()
    )
    return (a.data.storage_offset() == b.data.storage_offset() + allocate_size_b) or (
        b.data.storage_offset() == a.data.storage_offset() + allocate_size_a
    )


def group_continuous_tensors(tensors: List[torch.Tensor]):
    tensor_list = zip(tensors, list(range(len(tensors))))
    sorted_tensor_list = sorted(tensor_list, key=lambda x: x[0].storage_offset())

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


def calculate_mutual_groups(tensors_list: List[List[torch.Tensor]], op):
    constraints = []

    size = len(tensors_list[0])
    for tensors in tensors_list:
        assert size == len(
            tensors
        ), "Tensors to calculate mutual groups must have equal size."

        grouped_indices = group_continuous_tensors(tensors)

        if len(grouped_indices) > 1:
            constraints.append(grouped_indices)

    print(f"calculate mutual group, constraints={constraints}, op={op}")
    # no constraints, group them all
    if len(constraints) == 0:
        return [list(range(size))] if size > 0 else constraints

    grouped_indices = constraints[0]
    for i in range(1, len(constraints)):
        grouped_indices = op(grouped_indices, constraints[i])

    return grouped_indices


def intersect(a: List[List[int]], b: List[List[int]]):
    c = [value for value in a if value in b]
    return c


def union(a: List[List[int]], b: List[List[int]]):
    c1 = [value for value in a]
    c2 = [value for value in b if value not in a]

    return c1 + c2


def difference(a: List[List[int]], b: List[List[int]]):
    c = [value for value in a if value not in b]
    return c


def check_duplicated(values_list: List[List[int]]):
    values_list_flat = []

    for values in values_list:
        for v in values:
            if v in values_list_flat:
                return True
            values_list_flat.append(v)
    return False


def collocate_tensors(tensors: List[torch.Tensor], grouped_indices: List[List[int]]):
    tensor_map = {idx: tensor for idx, tensor in enumerate(tensors)}

    colocated = []
    for indices in grouped_indices:
        start = -1
        offset = 0
        for i in indices:
            tensor = tensor_map[i]
            if start == -1:
                start = tensor.storage_offset()

            assert (
                start + offset == tensor.storage_offset()
            ), "tensors collocated must be contiguous"

            offset += (
                tensor.bagua_tensor.num_elem_allocated()
                if hasattr(tensor, "bagua_tensor")
                else tensor.numel()
            )

        with torch.no_grad():
            tensor_view = torch.zeros(offset, dtype=tensors[0].dtype).to(
                tensors[0].device
            )
            tensor_view.set_(tensors[0].data.storage(), start, tensor_view.shape)

            colocated.append(tensor_view)

    return colocated


def fuse_optimizer(optimizer, do_flatten: bool = False):
    """Convert any optimizer into a fused optimizer.

    This fused optimizer fuses multiple module parameter update kernel launches
    into one or a few, by flattening parameter tensors into one or more
    contiguous buckets.

    It can be used in conjunction with :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua` method. In this case,
    Bagua will do the fusions automatically, otherwise, you need to explicitly
    set :attr:`do_flatten=True`.

    Args:
        optimizer (torch.optim.Optimizer): Any PyTorch optimizer.
        do_flatten (bool): Whether to flatten the parameters. Default: ``False``.

    Returns:
        Fused optimizer.


    Example::
        To use in conjunction with :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua` method:

        >>> optimizer = torch.optim.Adadelta(model.parameters(), ....)
        >>> optimizer = bagua.torch_api.contrib.fuse_optimizer(optimizer)
        >>> model = model.with_bagua([optimizer], GradientAllReduceAlgorithm())

        To use alone or with `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel>`_,
        set :attr:`do_flatten=True`:

        >>> optimizer = torch.optim.Adadelta(model.parameters(), ....)
        >>> optimizer = bagua.torch_api.contrib.fuse_optimizer(optimizer, do_flatten=True)
    """

    fused_optimizer = copy.copy(optimizer)

    # FIXME
    fused_optimizer.step_counter = 0
    optimizer._fused_optimizer = fused_optimizer

    if do_flatten:
        flatten_param_and_states(optimizer)

    if not hasattr(optimizer, "fuse_step"):
        patch = gorilla.Patch(optimizer.__class__, "fuse_step", fuse_step)
        gorilla.apply(patch)

    return optimizer


def is_fused_optimizer(optimizer: torch.optim.Optimizer):
    return hasattr(optimizer, "fuse_step")


def fuse_step(optimizer: torch.optim.Optimizer, closure=None):
    r"""Performs a single optimization step (parameter update).

    Args:
        closure (Callable): A closure that reevaluates the model and
            returns the loss. Optional for most optimizers.

    .. note::
        Unless otherwise specified, this function should not modify the
        ``.grad`` field of the parameters.
    """
    assert hasattr(
        optimizer, "_fused_optimizer"
    ), "Should init fused optimizer by calling `fuse_optimizer`."

    optimizer._fused_optimizer.step_counter += 1
    do_fuse(optimizer._fused_optimizer)
    return optimizer._fused_optimizer.step(closure)


def do_fuse(optimizer: torch.optim.Optimizer):
    for index, group in enumerate(optimizer.param_groups):
        params = group["params"]

        weights = [p.data for p in params]
        grads = [p.grad for p in params]

        state_tensors, state_scalars, succ = _get_state_by_name(optimizer.state, params)

        if not succ:
            continue

        grouped_indices = calculate_mutual_groups(
            [weights, grads] + list(state_tensors.values()), op=intersect
        )

        print("grouped indices: ", grouped_indices)

        if len(grouped_indices) > 0:
            # collocate params
            collocated_weights = collocate_tensors(weights, grouped_indices)
            collocated_grads = collocate_tensors(grads, grouped_indices)

            collocated_states = {}
            for name, tensors in state_tensors.items():
                ts = collocate_tensors(tensors, grouped_indices)
                collocated_states[name] = ts

            new_params = []
            for i in range(len(collocated_weights)):
                with torch.no_grad():
                    p = torch.nn.Parameter(collocated_weights[i], requires_grad=False)
                    p.grad = collocated_grads[i]

                new_params.append(p)

                for name, ts in collocated_states.items():
                    optimizer.state[p][name] = ts[i]

                for name, v in state_scalars.items():
                    optimizer.state[p][name] = v

            # add other params and remove dup states
            grouped_indices_flat = list(reduce(lambda x, y: x + y, grouped_indices))
            for idx, param in enumerate(params):
                if idx not in grouped_indices_flat:
                    new_params.append(param)
                    del optimizer.state[param]

            group["params"] = new_params
            print(
                f"Final at step #{optimizer.step_counter}, param_groups: {optimizer.param_groups}, states: {optimizer.state}"
            )


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

        print(f"state tensors: {state_tensors}, state scalars: {state_scalars}")

        for name, tensors in state_tensors.items():
            if len(tensors) != len(params):
                logging.error(
                    f"Parameter state '{name}' does not match, could not fuse optimizer."
                )
                return None, None, False

    return state_tensors, state_scalars, True
