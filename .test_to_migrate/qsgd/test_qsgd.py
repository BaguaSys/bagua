#!/usr/bin/env python3

import torch
from utils import qsgd_compress, qsgd_decompress
import numpy as np
import cupy


def test_qsgd_compression():
    t = torch.tensor([0.1, 0.5, 1.5, -0.5, 2.4]).cuda()
    norm, sign, cupy_compressed_ints = qsgd_compress(t, pnorm=2.0)
    assert norm.item() == 2.9189038276672363
    assert sign.item() == 232
    assert (
        cupy.asnumpy(cupy_compressed_ints) - np.array([9, 44, 132, 44, 211]) <= 1
    ).all()


def test_qsgd_decompression():
    t = torch.tensor([0.1, 0.5, 1.5, -0.5, 2.4]).cuda()
    norm, sign, cupy_compressed_ints = qsgd_compress(t, pnorm=2.0)
    t_decompressed = qsgd_decompress(norm, sign, cupy_compressed_ints)
    assert ((t_decompressed - t).abs() <= 1 / 256 * norm).all()


if __name__ == "__main__":
    cupy.random.seed(42)
    torch.manual_seed(42)
    cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    cupy_stream.use()
    test_qsgd_compression()
    test_qsgd_decompression()
