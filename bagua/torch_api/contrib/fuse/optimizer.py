import torch
from typing import List
import copy
import logging
from functools import reduce


def get_flattened_tensor(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 0:
        return

    total_size = 0
    for tensor in tensors:
        total_size += tensor.numel()

    flatten_tensor = torch.zeros(total_size, dtype=tensors[0].dtype).to(
        tensors[0].device
    )

    offset = 0
    for tensor in tensors:
        # copy data
        flatten_tensor[offset : offset + tensor.numel()] = tensor.data.reshape(-1)
        offset += tensor.numel()

    return flatten_tensor


def flatten_module_param_groups(optimizer: torch.optim.Optimizer):
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
            grads = [
                p.grad if p.grad is not None else torch.zeros_like(p.data)
                for p in params
            ]

            flattened_weight = get_flattened_tensor(weights)
            flattened_grad = get_flattened_tensor(grads)

            print(
                f"{param_type} before flattened: weight={weights}, after flattened: weight={flattened_weight}"
            )
            print(
                f"{param_type} before flattened: grads={grads}, after flattened: weight={flattened_grad}"
            )

            def set_storage(param, weight_storage, grad_storage, storage_offset):
                with torch.no_grad():
                    z = torch.zeros_like(param.data)
                    z.set_(weight_storage, storage_offset, param.shape)
                    param.data = z

                    t = torch.zeros_like(param.data)
                    t.set_(grad_storage, storage_offset, param.shape)
                    param.grad = t

            offset = 0
            for p in params:
                set_storage(
                    p, flattened_weight.storage(), flattened_grad.storage(), offset
                )
                offset += p.numel()

                print(f"flatten param done {offset}")

            weights = [p.data for p in params]
            grads = [p.grad for p in params]
            assert check_contiguous(weights)
            assert check_contiguous(grads)


def check_contiguous(tensors: List[torch.Tensor]) -> bool:
    data_ptr = None
    for t in tensors:
        if data_ptr is not None and t.data_ptr() != data_ptr:
            return False
        data_ptr = t.data_ptr() + t.numel() * t.element_size()
    return True


def _is_contiguous_tensor(a: torch.Tensor, b: torch.Tensor):
    allocate_size_a = (
        a.bagua_tensor.num_elem_allocated() if hasattr(a, "bagua_tensor") else a.numel()
    )
    allocate_size_b = (
        b.bagua_tensor.num_elem_allocated() if hasattr(b, "bagua_tensor") else b.numel()
    )
    return (a.data.storage_offset() == b.data.storage_offset() + allocate_size_b) or (
        b.data.storage_offset() == a.data.storage_offset() + allocate_size_a
    )


def _group_continuous_tensors(tensors: List[torch.Tensor]):
    tensor_list = zip(tensors, list(range(len(tensors))))
    sorted_tensor_list = sorted(tensor_list, key=lambda x: x[0].storage_offset())

    grouped = []
    tmp_tensors = []
    tmp_indices = []

    for tensor, idx in sorted_tensor_list:
        if len(tmp_tensors) > 0 and not _is_contiguous_tensor(tensor, tmp_tensors[-1]):
            if len(tmp_tensors) > 1:
                grouped.append(tmp_indices)
            tmp_tensors = []
            tmp_indices = []

        tmp_tensors.append(tensor)
        tmp_indices.append(idx)

    if len(tmp_tensors) > 1:
        grouped.append(tmp_indices)

    return grouped


def _get_intersection(a: List[List[int]], b: List[List[int]]):
    c = [value for value in a if value in b]
    return c


def _collocate(tensors: List[torch.Tensor], grouped_indices: List[List[int]]):
    tensor_map = {idx: tensor for idx, tensor in enumerate(tensors)}

    colocated_tensors = []
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

            colocated_tensors.append(tensor_view)

    return colocated_tensors


class FusedOptimizer(torch.optim.Optimizer):
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
        >>> optimizer = bagua.torch_api.contrib.FusedOptimizer(optimizer)
        >>> model = model.with_bagua([optimizer], GradientAllReduceAlgorithm())

        To use alone or with `torch.nn.parallel.DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel>`_,
        set :attr:`do_flatten=True`:

        >>> optimizer = torch.optim.Adadelta(model.parameters(), ....)
        >>> optimizer = bagua.torch_api.contrib.FusedOptimizer(optimizer, do_flatten=True)
    """

    def __init__(self, optimizer: torch.optim.Optimizer, do_flatten: bool = False):
        self.optimizer = copy.copy(optimizer)
        super(FusedOptimizer, self).__init__(optimizer.param_groups, optimizer.defaults)

        if do_flatten:
            flatten_module_param_groups(optimizer)

        self.step_counter = 0

    def step(self, closure=None):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        self.step_counter += 1
        self.fuse()
        return self.optimizer.step(closure)

    def fuse(self):
        for index, group in enumerate(self.optimizer.param_groups):
            params = group["params"]

            weights = [p.data for p in params]
            grads = [p.grad for p in params]

            grouped_weight_indices = _group_continuous_tensors(weights)
            grouped_grad_indices = _group_continuous_tensors(grads)

            grouped_indices = _get_intersection(
                grouped_weight_indices, grouped_grad_indices
            )

            print(
                f"step #{self.step_counter}: grouped indices: {grouped_weight_indices}, {grouped_grad_indices}"
            )

            if len(grouped_indices) == 0:
                break

            state_tensors = {}
            state_scalars = {}

            if len(self.optimizer.state) > 0:
                state_tensors = {
                    name: []
                    for name, value in self.optimizer.state[params[0]].items()
                    if isinstance(value, torch.Tensor)
                }
                state_scalars = {
                    name: value
                    for name, value in self.optimizer.state[params[0]].items()
                    if not isinstance(value, torch.Tensor)
                }

                for p in params:
                    st = self.optimizer.state[p]

                    for name, value in st.items():
                        if isinstance(value, torch.Tensor):
                            if state_tensors.get(name) is None:
                                logging.error(
                                    f"Unexpected tensor in state {name}, could not fuse optimizer."
                                )
                                return

                            state_tensors[name].append(value)
                        else:
                            if state_scalars.get(name) is None:
                                logging.error(
                                    f"Unexpected scalar value in state {name}, could not fuse optimizer."
                                )
                                return

                            if value != state_scalars[name]:
                                logging.error(
                                    f"Parameter state '{name}' does not match, could not fuse optimizer."
                                )
                                return

                print(f"state tensors: {state_tensors}, state scalars: {state_scalars}")

                for name, tensors in state_tensors.items():
                    if len(tensors) != len(params):
                        logging.error(
                            f"Parameter state '{name}' does not match, could not fuse optimizer."
                        )
                        return

                for name, tensors in state_tensors.items():
                    indices = _group_continuous_tensors(tensors)
                    grouped_indices = _get_intersection(grouped_indices, indices)
                    print(
                        f"step #{self.step_counter}: state: {name}, indices: {indices}, grouped_indices: {grouped_indices}"
                    )

            if len(grouped_indices) > 0:
                # collocate params
                collocated_weights = _collocate(weights, grouped_indices)
                collocated_grads = _collocate(grads, grouped_indices)

                collocated_states = {}
                for name, tensors in state_tensors.items():
                    ts = _collocate(tensors, grouped_indices)
                    collocated_states[name] = ts

                new_params = []
                for i in range(len(collocated_weights)):
                    with torch.no_grad():
                        p = torch.nn.Parameter(
                            collocated_weights[i], requires_grad=False
                        )
                        p.grad = collocated_grads[i]

                    new_params.append(p)

                    for name, ts in collocated_states.items():
                        self.optimizer.state[p][name] = ts[i]

                    for name, v in state_scalars.items():
                        self.optimizer.state[p][name] = v

                # add other params and remove dup states
                grouped_indices_flat = list(reduce(lambda x, y: x + y, grouped_indices))
                for idx, param in enumerate(params):
                    if idx not in grouped_indices_flat:
                        new_params.append(param)
                        del self.optimizer.state[param]

                group["params"] = new_params
                print(
                    f"Final at step #{self.step_counter}, param_groups: {self.optimizer.param_groups}, states: {self.optimizer.state}"
                )
