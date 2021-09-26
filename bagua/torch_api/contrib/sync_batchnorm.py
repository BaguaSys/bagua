# Copyright (c) Uber Technologies, Inc. and its affiliates.
# Copyright (c) 2021 Kuaishou AI Platform & DS3 Lab.
#
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from distutils.version import LooseVersion

import torch
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import bagua.torch_api as bagua
from bagua.torch_api.communication import allgather, allreduce

# Backward compat for old PyTorch
if not hasattr(torch.jit, "unused"):
    torch.jit.unused = lambda x: x


_SYNC_BN_V2 = LooseVersion(torch.__version__) >= LooseVersion("1.5.0") and LooseVersion(
    torch.__version__
) <= LooseVersion("1.6.0")
_SYNC_BN_V3 = LooseVersion(torch.__version__) >= LooseVersion("1.6.0")
_SYNC_BN_V4 = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class SyncBatchNorm(_BatchNorm):
    r"""Applies synchronous BatchNorm for distributed module with N-dimensional BatchNorm layer(s).
    See `BatchNorm <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html?highlight=batchnorm#torch.nn.BatchNorm2d>`_ for more details.

    Arguments:
        num_features: Number of channels :math:`C` from the shape :math:`(N, C, ...)`.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.
        momentum: The value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1.
        affine: A boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``.
        track_running_stats: A boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``.

    .. note:: Only GPU input tensors are supported in the training mode.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError(
                "expected at least 2D input (got {}D input)".format(input.dim())
            )

    def _run_bn(self, input):
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )

    @torch.jit.unused
    def _maybe_run_sync_bn(self, input):
        if bagua.get_world_size() == 1:
            return self._run_bn(input)
        return _SyncBatchNorm.apply(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
        )

    def forward(self, input):
        # currently only GPU input is supported by underlying kernel from PyTorch
        if not input.is_cuda:
            raise ValueError("SyncBatchNorm expected input tensor to be on GPU")

        self._check_input_dim(input)

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked = self.num_batches_tracked + 1

        if not self.training and self.track_running_stats:
            return self._run_bn(input)
        else:
            return self._maybe_run_sync_bn(input)

    @classmethod
    def convert_sync_batchnorm(cls, module):
        r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        `torch.nn.SyncBatchNorm <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html?highlight=syncbatchnorm#torch.nn.SyncBatchNorm>`_ layers.

        Arguments:
            module (nn.Module): Module containing one or more :attr:`BatchNorm*D` layers

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        .. note:: This function must be called before :meth:`~bagua.torch_api.distributed.BaguaModule.with_bagua` method.

        Example::
            >>> # Network with nn.BatchNorm layer
            >>> model = torch.nn.Sequential(
            ...      torch.nn.Linear(D_in, H),
            ...      torch.nn.ReLU(),
            ...      torch.nn.Linear(H, D_out),
            ...    )
            >>> optimizer = torch.optim.SGD(
            ...      model.parameters(),
            ...      lr=0.01,
            ...      momentum=0.9
            ...    )
            >>> sync_bn_model = bagua.torch_api.contrib.sync_batchnorm.SyncBatchNorm.convert_sync_batchnorm(model)
            >>> bagua_model = sync_bn_model.with_bagua([optimizer], GradientAllReduce())
        """
        module_output = module

        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child))
        del module
        return module_output


class _SyncBatchNorm(Function):
    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum):
        input = input.contiguous()

        size = input.numel() // input.size(1)
        count = torch.tensor([size])

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)
        count, mean, invstd = count.cuda(), mean.cuda(), invstd.cuda()

        nums_ranks = bagua.get_world_size()
        count_all = torch.tensor(
            [torch.empty_like(count).cpu().detach().numpy() for _ in range(nums_ranks)]
        ).cuda()
        mean_all = torch.tensor(
            [torch.empty_like(mean).cpu().detach().numpy() for _ in range(nums_ranks)]
        ).cuda()
        invstd_all = torch.tensor(
            [torch.empty_like(invstd).cpu().detach().numpy() for _ in range(nums_ranks)]
        ).cuda()

        allgather(count.unsqueeze(0), count_all)
        allgather(mean.unsqueeze(0), mean_all)
        allgather(invstd.unsqueeze(0), invstd_all)

        if _SYNC_BN_V3:
            counts_for_bngswc = count_all.view(-1).float().to(input.device)
        else:
            # backwards compatibility
            counts_for_bngswc = count_all.view(-1).tolist()

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            counts_for_bngswc,
        )

        self.save_for_backward(input, weight, mean, invstd, count_all)

        # apply element-wise normalization
        return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_all = self.saved_tensors
        need_input_grad, need_weight_grad, need_bias_grad = self.needs_input_grad[0:3]

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            need_input_grad,
            need_weight_grad,
            need_bias_grad,
        )

        if need_input_grad:
            # synchronizing stats used to calculate input gradient.
            allreduce(sum_dy, sum_dy)
            allreduce(sum_dy_xmu, sum_dy_xmu)

            if _SYNC_BN_V4:
                # from 1.9.0 on we need a count tensor on all devices
                # count_all is calculated as total count across all ranks in forward function
                count_all = count_all.to(dtype=torch.int, device=grad_output.device)
            elif _SYNC_BN_V2 or _SYNC_BN_V3:
                # before 1.9.0 we need the count as an integer to compute means values
                count = count_all.sum()
            else:
                # before 1.5.0, sum_dy was sum of means from every worker, so we just
                # need to divide it by number of workers
                count = bagua.get_world_size()

            # backward pass for gradient calculation
            # we are calling into a non-public undocumented function which broke moving to 1.9.0
            # https://github.com/pytorch/pytorch/issues/57900
            if _SYNC_BN_V4:
                # from 1.9.0 on, sums and count parameters expected
                grad_input = torch.batch_norm_backward_elemt(
                    grad_output,
                    saved_input,
                    mean,
                    invstd,
                    weight,
                    sum_dy,
                    sum_dy_xmu,
                    count_all,
                )
            else:
                # before 1.9.0, mean parameters expected, not sums and count
                grad_input = torch.batch_norm_backward_elemt(
                    grad_output,
                    saved_input,
                    mean,
                    invstd,
                    weight,
                    sum_dy / count,
                    sum_dy_xmu / count,
                )
        else:
            grad_input = None

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not need_weight_grad:
            grad_weight = None

        if weight is None or not need_bias_grad:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None
