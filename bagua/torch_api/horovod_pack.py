import torch
import collections
from collections.abc import Iterable
from .communication import broadcast_coalesced
from .env import get_rank
import logging


def initialize_optimizer_state_0_21_1(optimizer):
    """
    Broadcasts an optimizer state from root rank to all other processes.
    Arguments:
            optimizer: An optimizer.
            root_rank: The rank of the process from which the optimizer will be
                               broadcasted to all other processes.
    """
    if get_rank() == 0:
        logging.info("Horovod v0.21.1 compat mode enabled.")

    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        #  the entire state_dict, as its structure is deeply nested and contains
        #  None type parameter values
        raise ValueError("cannot broadcast torch.optim.LBFGS state")

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict["state"]) == 0:
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad and id(p) not in state_dict["state"]:
                    p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict["state"]) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict["state"][pid][name] = t(p.cpu().numpy()[0])

        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(
                option_tensor.cpu().numpy()[0], dtypes
            )

        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict["param_groups"]):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == "params":
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = "%s.%d" % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(
                index, option_key, option_tensor, dtypes
            )
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group["params"]:
            if pid not in state_dict["state"]:
                # The param has not set requires_grad, so skip broadcast
                continue

            param_state = state_dict["state"][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = "%s.%d" % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Post-broadcast cleanup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()


# Reference: https://github.com/horovod/horovod/blob/v0.21.3/horovod/torch/functions.py#L61
def initialize_optimizer_state_0_21_3(optimizer):
    if get_rank() == 0:
        logging.info("Horovod v0.21.3 compat mode enabled.")

    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        #  the entire state_dict, as its structure is deeply nested and contains
        #  None type parameter values
        raise ValueError("cannot broadcast torch.optim.LBFGS state")

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict["state"]) == 0:
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad and id(p) not in state_dict["state"]:
                    p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        optimizer.step()
