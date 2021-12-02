import copy
import math
import os
import random
import signal
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from datetime import timedelta
from itertools import product
from unittest import mock

import torch
import torch.distributed as c10d

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    requires_nccl_version,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    retry_on_connect_failures,
    TEST_WITH_TSAN,
)
from . import test_c10d_common
from .test_c10d_common import gpus_for_rank, DoubleGpuNet, ConvNet, ModuleForDdpCommHook

import bagua.torch_api as bagua
from bagua.torch_api.data_parallel.distributed import DistributedDataParallel_V1_9_0 as DistributedDataParallel


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class DistributedDataParallelTest(test_c10d_common.AbstractDistributedDataParallelTest, MultiProcessTestCase):

    def setUp(self):
        super(DistributedDataParallelTest, self).setUp()
        # NCCL_BLOCKING_WAIT overrides NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def _test_find_unused_parameters_kwarg(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}"
        )
        bagua.init_process_group()
        process_group = c10d.distributed_c10d._get_default_group()

        class FindUnusedParametersModule(nn.Module):
            def __init__(self):
                super(FindUnusedParametersModule, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # Return the fc3 module so that the caller can invoke it
                # outside of the forward function. While this is bad practice,
                # we can use it to trigger a reducer error.
                return (F.softmax(x, dim=1), self.fc3)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        ddp_model = None

        def test_find_unused_parameters(
                find_unused_parameters, test_default=False, gradient_as_bucket_view=False
        ):
            if test_default:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )
            else:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    find_unused_parameters=find_unused_parameters,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )
            nonlocal ddp_model
            ddp_model = model

            output, fc3 = model(input)
            output = fc3(output)
            loss = criterion(output, target)
            loss.backward()

        # First test that finding unused params under these conditions is to
        # trigger an error when `backward` is called (because fc3 is an unused
        # parameter and will therefore be marked ready twice).
        try:
            test_find_unused_parameters(
                True, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.assertTrue(
                str(ex).startswith(
                    "Expected to mark a variable ready only once.",
                )
            )
            unused_index = 2
            unused_index_str = f"Parameter at index {unused_index}"
            model = ddp_model.module
            for module_name, module in model.named_modules():
                if module == model.fc3:
                    for parameter_name, _ in module.named_parameters(
                            recurse=False
                    ):
                        unused_fqn = f"{module_name}.{parameter_name}"
                        # Only one such parameter in model.fc3, since bias=False
                        break

            if dist._get_debug_mode() != dist._DistributedDebugLevel.OFF:
                unused_index_str += f" with name {unused_fqn}"

            self.assertTrue(unused_index_str in str(ex))
        else:
            self.fail("Expected exception")

        dist.barrier(process_group)

        # Then test that the default behavior can be overridden by setting
        # `find_unused_parameters=False`.
        try:
            test_find_unused_parameters(
                False, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.fail("Unexpected exception: %s" % ex)

        # Test find_unused_parameters defaults to False
        try:
            test_find_unused_parameters(
                True, test_default=True, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.fail("Unexpected exception: %s" % ex)

    # TODO: Combine the following tests once https://github.com/pytorch/pytorch/issues/55967
    # is resolved.
    @skip_if_lt_x_gpu(2)
    def test_find_unused_parameters_kwarg_debug_detail(self):
        self._test_find_unused_parameters_kwarg()

    def _test_multiple_outputs_multiple_backward(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class MultipleOutputModule(nn.Module):
            def __init__(self):
                super(MultipleOutputModule, self).__init__()

                def define_module():
                    return nn.Sequential(
                        nn.Linear(2, 10, bias=False),
                        nn.ReLU(),
                        nn.Linear(10, 4, bias=False),
                        nn.ReLU(),
                    )

                self.module0 = define_module()
                self.module1 = define_module()

            def forward(self, x):
                return (
                    F.softmax(self.module0(x), dim=1),
                    F.softmax(self.module1(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            MultipleOutputModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # Compute loss and gradients for both outputs
        output1, output2 = model(input)
        loss1 = criterion(output1, target)
        loss1.backward()
        loss2 = criterion(output2, target)
        loss2.backward()

    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward(self):
        self._test_multiple_outputs_multiple_backward()

    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward_grad_is_view(self):
        self._test_multiple_outputs_multiple_backward(gradient_as_bucket_view=True)


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
