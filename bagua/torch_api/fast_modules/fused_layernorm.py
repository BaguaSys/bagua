# Copyright (c) 2021 Kuaishou AI Platform & DS3 Lab. All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton layer norm tutorial
# https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py

import torch
import torch.nn as nn
import numbers
import triton
import triton.language as tl


# Forward Pass
@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, M, V, stride, N, eps, **META):
    BLOCK_SIZE = META["BLOCK_SIZE"]
    # position of elements processed by this program
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    # offset data pointers to start at the row of interest
    X += row * stride
    Y += row * stride
    # load data and cast to float32
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    # compute mean
    mean = tl.sum(x, axis=0) / N
    # compute std
    xmean = tl.where(mask, x - mean, 0.)
    var = tl.sum(xmean * xmean, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    xhat = xmean * rstd
    # write-back mean/rstd
    tl.store(M + row, mean)
    tl.store(V + row, rstd)
    # multiply by weight and add bias
    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    y = xhat * w + b
    # write-back
    tl.store(Y + cols, y, mask=mask)


# Backward pass (DX + partial DW + partial DB)
@triton.jit
def _layer_norm_bwd_dx_fused(DX, DY, DW, DB, X, W, B, M, V, Lock, stride, N, eps, **META):
    GROUP_SIZE_M = META["GROUP_SIZE_M"]
    BLOCK_SIZE_N = META["BLOCK_SIZE_N"]
    # position of elements processed by this program
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    # offset data pointers to start at the row of interest
    X += row * stride
    DY += row * stride
    DX += row * stride
    # offset locks and weight/bias gradient pointer
    # each kernel instance accumulates partial sums for
    # DW and DB into one of GROUP_SIZE_M independent buffers
    # these buffers stay in the L2, which allow this kernel
    # to be fast
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(M + row)
    rstd = tl.load(V + row)
    # compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * mean1 + mean2)) * rstd
    # write-back dx
    tl.store(DX + cols, dx, mask=mask)
    # accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # first store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    # release lock
    tl.atomic_xchg(Lock, 0)


# Backward pass (total DW + total DB)
@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N, **meta):
    pid = tl.program_id(0)
    BLOCK_SIZE_M = meta["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = meta["BLOCK_SIZE_N"]
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, meta["BLOCK_SIZE_M"])
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M,)](x_arg, y, weight, bias, mean, rstd,
                                    x_arg.stride(0), N, eps,
                                    BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DG/DB
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
        _dw = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device)
        _db = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device)
        dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        db = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M,)](dx, dy, _dw, _db, x, w, b, m, v, locks,
                                       x_arg.stride(0), N, ctx.eps,
                                       BLOCK_SIZE_N=ctx.BLOCK_SIZE,
                                       GROUP_SIZE_M=GROUP_SIZE_M,
                                       num_warps=ctx.num_warps)

        def triton_cdiv(N):
            def make_cdiv(meta):
                return [triton.cdiv(N, meta["BLOCK_SIZE_N"])]
            return make_cdiv
        grid = triton_cdiv(N)
        # accumulate partial sums in separate kernel
        _layer_norm_bwd_dwdb[grid](_dw, _db, dw, db, GROUP_SIZE_M, N, BLOCK_SIZE_M=32, BLOCK_SIZE_N=128)
        return dx, None, dw, db, None


class FusedLayerNorm(nn.Module):
    """
    Handle a layer normalization, like torch.nn.LayerNorm_.
    This implementation should be measurably faster than the default PyTorch layernorm (as of PyTorch 1.9),
    both for training and inference worloads.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super(FusedLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, device="cuda"))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device="cuda"))

    def forward(self, x):
        return _LayerNorm.apply(x, self.normalized_shape, self.weight, self.bias, self.eps)
