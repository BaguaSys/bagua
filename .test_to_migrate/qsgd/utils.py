#!/usr/bin/env python3
"""
Utilities for QSGD algorithm.
"""
from torch.distributions.bernoulli import Bernoulli
import torch
import cupy
from bagua.torch_api.utils import torch2cupy, cupy2torch


def qsgd_compress(t, pnorm="inf", quan_bits=8):
    """
    Quantize float32 into uint8 using QSGD algorithm.

    Arguments:
        * `t` - Input PyTorch CUDA Tensor to be quantized.
        * `pnorm` - Order of norm used for QSGD compression. Default value is `inf`.
        * `quan_bits` - Number of quantization bits used for QSGD compression. Default
           value is `8`. Other values like `1`,`2`,`4` are not supported at present.

    Yields:
        (Torch.tensor, Torch.tensor, Torch.tensor):  Quantized PyTorch CUDA Tensors.

    Examples:

    ```python
    cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    cupy_stream.use()
    norm, sign, compressed_ints = qsgd_compress(tensor)
    ```

    ..note::
        CuPy and PyTorch use different default CUDA streams. Force CuPy to use PyTorch
        current CUDA stream to simplify stream synchronization.

    """
    _quantization_level = (1 << quan_bits) - 1

    norm = t.norm(float(pnorm))

    sign = (torch.sign(t) + 1.0).bool()

    _level = t / norm * _quantization_level
    _bernoulli_probs = _level - _level.floor()
    _incr = Bernoulli(probs=_bernoulli_probs).sample()
    _compressed_floats = (_level.floor() + _incr.float()).abs()
    compressed_floats = torch.clamp(_compressed_floats, max=_quantization_level)
    compressed_ints = compressed_floats.byte()

    packed_sign = _cupy_packbits(sign)
    return norm, packed_sign, compressed_ints


def qsgd_decompress(norm, packed_sign, compressed_ints, quan_bits=8):
    """
    The reverse of the ``qsgd_compress`` function.

    Arguments:
        * `norm` - Order of norm used in QSGD compression.
        * `packed_sign` - Packed sign of quantized tensor.
        * `compressed_ints` - Absolute value of quantized tensor.
        * `quan_bits` - Number of quantization bits. Default value is 8. Other
           values like `1`,`2`,`4` is not supported at present.

    Yields:
        Torch.tensor: De-quantized tensor.

    Examples:

    ```python
    cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    cupy_stream.use()
    tensor = qsgd_compress(norm, sign, compressed_ints)
    ```

    ..note::
        CuPy and PyTorch use different default CUDA streams. Force CuPy to use PyTorch
        current CUDA stream to simplify stream synchronization.
    """

    compressed_floats = compressed_ints.float()
    _quantization_level = (1 << quan_bits) - 1
    sign = _cupy_unpackbits(packed_sign, compressed_floats.size(0))
    return norm * (sign.float() * 2 - 1) * compressed_floats / _quantization_level


def _cupy_packbits(tensor):
    cupy_tensor = torch2cupy(tensor)
    packed_cupy_tensor = cupy.packbits(cupy_tensor)
    packed_tensor = cupy2torch(packed_cupy_tensor)
    return packed_tensor


def _cupy_unpackbits(tensor, size):
    cupy_tensor = torch2cupy(tensor)
    unpacked_cupy_tensor = cupy.unpackbits(cupy_tensor)[0:size]
    tensor = cupy2torch(unpacked_cupy_tensor)
    return tensor
