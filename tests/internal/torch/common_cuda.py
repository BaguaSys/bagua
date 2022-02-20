# From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
# and IDIAP Research Institute nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# This file is copied from https://github.com/pytorch/pytorch/tree/v1.9.0/torch/testing/_internal/common_cuda.py
r"""This file is allowed to initialize CUDA context when imported."""

import functools
import torch
import torch.cuda
from tests.internal.torch.common_utils import TEST_NUMBA
import inspect
import contextlib
from setuptools import distutils


TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
CUDA_DEVICE = torch.device("cuda:0") if TEST_CUDA else None
# note: if ROCm is targeted, TEST_CUDNN is code for TEST_MIOPEN
TEST_CUDNN = TEST_CUDA and torch.backends.cudnn.is_acceptable(
    torch.tensor(1.0, device=CUDA_DEVICE)
)
TEST_CUDNN_VERSION = torch.backends.cudnn.version() if TEST_CUDNN else 0

CUDA11OrLater = (
    torch.version.cuda
    and distutils.version.LooseVersion(torch.version.cuda) >= "11.0.0"
)
CUDA9 = torch.version.cuda and torch.version.cuda.startswith("9.")
SM53OrLater = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (5, 3)

TEST_MAGMA = TEST_CUDA
if TEST_CUDA:
    torch.ones(1).cuda()  # has_magma shows up after cuda is initialized
    TEST_MAGMA = torch.cuda.has_magma

if TEST_NUMBA:
    import numba.cuda

    TEST_NUMBA_CUDA = numba.cuda.is_available()
else:
    TEST_NUMBA_CUDA = False

# Used below in `initialize_cuda_context_rng` to ensure that CUDA context and
# RNG have been initialized.
__cuda_ctx_rng_initialized = False


# after this call, CUDA context and RNG must have been initialized on each GPU
def initialize_cuda_context_rng():
    global __cuda_ctx_rng_initialized
    assert TEST_CUDA, "CUDA must be available when calling initialize_cuda_context_rng"
    if not __cuda_ctx_rng_initialized:
        # initialize cuda context and rng for memory tests
        for i in range(torch.cuda.device_count()):
            torch.randn(1, device="cuda:{}".format(i))
        __cuda_ctx_rng_initialized = True


# Test whether hardware TF32 math mode enabled. It is enabled only on:
# - CUDA >= 11
# - arch >= Ampere
def tf32_is_not_fp32():
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    if int(torch.version.cuda.split(".")[0]) < 11:
        return False
    return True


@contextlib.contextmanager
def tf32_off():
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=False
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul


@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_precison = self.precision
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        self.precision = tf32_precision
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=True
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul
        self.precision = old_precison


# This is a wrapper that wraps a test to run this test twice, one with
# allow_tf32=True, another with allow_tf32=False. When running with
# allow_tf32=True, it will use reduced precision as pecified by the
# argument. For example:
#    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
#    @tf32_on_and_off(0.005)
#    def test_matmul(self, device, dtype):
#        a = ...; b = ...;
#        c = torch.matmul(a, b)
#        self.assertEqual(c, expected)
# In the above example, when testing torch.float32 and torch.complex64 on CUDA
# on a CUDA >= 11 build on an >=Ampere architecture, the matmul will be running at
# TF32 mode and TF32 mode off, and on TF32 mode, the assertEqual will use reduced
# precision to check values.
#
# This decorator can be used for function with or without device/dtype, such as
# @tf32_on_and_off(0.005)
# def test_my_op(self)
# @tf32_on_and_off(0.005)
# def test_my_op(self, device)
# @tf32_on_and_off(0.005)
# def test_my_op(self, device, dtype)
# @tf32_on_and_off(0.005)
# def test_my_op(self, dtype)
# if neither device nor dtype is specified, it will check if the system has ampere device
# if device is specified, it will check if device is cuda
# if dtype is specified, it will check if dtype is float32 or complex64
# tf32 and fp32 are different only when all the three checks pass
def tf32_on_and_off(tf32_precision=1e-5):
    def with_tf32_disabled(self, function_call):
        with tf32_off():
            function_call()

    def with_tf32_enabled(self, function_call):
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(f):
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            for k, v in zip(arg_names, args):
                kwargs[k] = v
            cond = tf32_is_not_fp32()
            if "device" in kwargs:
                cond = cond and (torch.device(kwargs["device"]).type == "cuda")
            if "dtype" in kwargs:
                cond = cond and (kwargs["dtype"] in {torch.float32, torch.complex64})
            if cond:
                with_tf32_disabled(kwargs["self"], lambda: f(**kwargs))
                with_tf32_enabled(kwargs["self"], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped

    return wrapper


# This is a wrapper that wraps a test to run it with TF32 turned off.
# This wrapper is designed to be used when a test uses matmul or convolutions
# but the purpose of that test is not testing matmul or convolutions.
# Disabling TF32 will enforce torch.float tensors to be always computed
# at full precision.
def with_tf32_off(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        with tf32_off():
            return f(*args, **kwargs)

    return wrapped


def _get_torch_cuda_version():
    if torch.version.cuda is None:
        return (0, 0)
    cuda_version = str(torch.version.cuda)
    return tuple(int(x) for x in cuda_version.split("."))
