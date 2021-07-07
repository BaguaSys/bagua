import torch
from bagua.torch_api.fuse_optimizer import FusedOptimizer
import logging


logging.basicConfig(level=logging.DEBUG)


def _get_aligned_flatten_tensor(tensors):
    flatten = torch._utils._flatten_dense_tensors
    flatten_tensor = flatten(tensors)
    t = torch.zeros(
        int((flatten_tensor.numel() / 8 + 1) * 8), dtype=flatten_tensor.dtype
    ).to(flatten_tensor.device)
    t[0 : flatten_tensor.numel()] = flatten_tensor
    return t


def _init_params():
    torch.manual_seed(49)
    t1 = torch.rand((10, 10), requires_grad=True)
    t2 = torch.rand((10, 10), requires_grad=True)
    t3 = torch.add(t1, t2)

    t4 = torch.rand((10, 10), requires_grad=True)
    t5 = torch.mul(t3, t4)

    t6 = torch.rand((10, 10), requires_grad=True)
    t7 = torch.rand((10, 10), requires_grad=True)
    t8 = torch.mul(t6, t7)

    t9 = torch.add(t5, t8)

    loss = t9.sum()
    loss.backward()

    return [t1, t2, t4, t6, t7]


def _flatten_params(params):

    param_tensors = list(map(lambda x: x.data, params))

    flatten_tensor = _get_aligned_flatten_tensor(param_tensors)
    flatten_storage = flatten_tensor.storage()
    offset = 0

    for p in params:
        with torch.no_grad():
            p.set_(flatten_storage, offset, p.shape)

        offset += p.numel()
    return flatten_tensor


def _flatten_params_grad(params):
    grad_tensors = list(map(lambda x: x.grad.data, params))

    flatten_tensor = _get_aligned_flatten_tensor(grad_tensors)
    flatten_storage = flatten_tensor.storage()
    offset = 0

    for p in params:
        with torch.no_grad():
            t = torch.zeros_like(p.data)
            t.set_(flatten_storage, offset, p.shape)
            p.grad = t

        offset += p.numel()
    return flatten_tensor


def test_unfused():
    params = _init_params()
    params_group = [
        {"params": params[0:-2], "weight_decay": 0.0},
        {"params": params[-2:], "weight_decay": 0.8},
    ]

    optimizer = torch.optim.SGD(params_group, lr=0.001)

    optimizer.step()

    flattened = _flatten_params(params)
    logging.info(f"flattened param norm: {torch.norm(flattened)}")


def test_unflattened():
    params = _init_params()
    params_group = [
        {"params": params[0:-2], "weight_decay": 0.0},
        {"params": params[-2:], "weight_decay": 0.8},
    ]

    optimizer = torch.optim.SGD(params_group, lr=0.001)
    fused_optimzer = FusedOptimizer(optimizer)

    fused_optimzer.step()

    flattened = _flatten_params(params)
    logging.info(f"flattened param norm: {torch.norm(flattened)}")


def test_flatten_grads():
    params = _init_params()
    params_group = [
        {"params": params[0:-2], "weight_decay": 0.0},
        {"params": params[-2:], "weight_decay": 0.8},
    ]

    _flatten_params_grad(params)
    optimizer = torch.optim.SGD(params_group, lr=0.001)
    fused_optimzer = FusedOptimizer(optimizer)

    fused_optimzer.step()
    flattened = _flatten_params(params)
    logging.info(f"flattened param norm: {torch.norm(flattened)}")


def test_flatten_weights():
    params = _init_params()
    params_group = [
        {"params": params[0:-2], "weight_decay": 0.0},
        {"params": params[-2:], "weight_decay": 0.8},
    ]

    _flatten_params(params)
    optimizer = torch.optim.SGD(params_group, lr=0.001)
    fused_optimzer = FusedOptimizer(optimizer)

    fused_optimzer.step()
    flattened = _flatten_params(params)
    logging.info(f"flattened param norm: {torch.norm(flattened)}")


def test_partial_flatten():
    params = _init_params()
    params_group = [
        {"params": params[0:-2], "weight_decay": 0.0},
        {"params": params[-2:], "weight_decay": 0.8},
    ]

    _flatten_params(params[1:-2])
    _flatten_params_grad(params[1:-2])

    optimizer = torch.optim.SGD(params_group, lr=0.001)
    fused_optimzer = FusedOptimizer(optimizer)

    fused_optimzer.step()
    flattened = _flatten_params(params)
    logging.info(f"flattened param norm: {torch.norm(flattened)}")


def test_all_flatten():
    params = _init_params()
    params_group = [
        {"params": params[0:-2], "weight_decay": 0.0},
        {"params": params[-2:], "weight_decay": 0.8},
    ]

    _flatten_params(params)
    _flatten_params_grad(params)

    optimizer = torch.optim.SGD(params_group, lr=0.001)
    fused_optimzer = FusedOptimizer(optimizer)

    fused_optimzer.step()
    flattened = _flatten_params(params)
    logging.info(f"flattened param norm: {torch.norm(flattened)}")


if __name__ == "__main__":
    test_unfused()
    test_unflattened()
    test_flatten_grads()
    test_flatten_weights()
    test_partial_flatten()
    test_all_flatten()
