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
from bagua.torch_api.data_parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    requires_nccl_version,
    skip_if_lt_x_gpu,
    get_timeout,
    skip_if_rocm,
    with_dist_debug_levels,
    with_nccl_blocking_wait,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    retry_on_connect_failures,
    TEST_WITH_TSAN,
)
from . import test_c10d_common
from .test_c10d_common import gpus_for_rank, DoubleGpuNet, ConvNet, ModuleForDdpCommHook


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
        if sys.platform == "win32":
            self._spawn_processes()
        else:
            self._spawn_processes()
            # self._fork_processes()

    def _test_nccl_backend(
            self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_multi_device_ids_not_allowed(self):
        int_devices = list(range(torch.cuda.device_count()))
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        with self.assertRaisesRegex(ValueError, "device_ids can only be None or contain a single element."):
            self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_single_device_module_device_ids_None(self):
        self._test_nccl_backend(None, None)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_single_device_module_empty_device_ids(self):
        # This tests the backward compatibility of accepting an empty list as `device_ids`,
        # although we no longer document this in favor of the default value of `None`,
        # which is consistent with multi-device modules and CPU modules.
        self._test_nccl_backend(None, [])

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_multi_device_module_device_ids_None(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_1gpu_module_device_ids_integer_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_1gpu_module_device_ids_torch_device_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_2gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(8)
    def test_nccl_backend_4gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_ddp_multi_device_module_config(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]

        self.assertTrue(len(gpus) >= 2, "expecting at least 2 gpus per process")

        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        gpus = gpus[:2]
        model = DoubleGpuNet(gpus)

        with self.assertRaisesRegex(
                ValueError,
                "DistributedDataParallel device_ids and output_device arguments only work with "
                "single-device/multiple-device GPU modules or CPU modules",
        ):
            ddp_model = DistributedDataParallel(
                model, output_device=gpus[1], process_group=process_group
            )

        with self.assertRaisesRegex(ValueError, "device_ids can only be None or contain a single element."):
            ddp_model = DistributedDataParallel(
                model, device_ids=gpus, process_group=process_group
            )

        with self.assertRaisesRegex(
                ValueError, "input module must be on the same type of devices"
        ):
            model.fc1 = model.fc1.cpu()
            ddp_model = DistributedDataParallel(model, process_group=process_group)

        model = model.cpu()
        with self.assertRaisesRegex(ValueError, "device_ids can only be None or contain a single element."):
            ddp_model = DistributedDataParallel(
                model, device_ids=gpus, process_group=process_group
            )

    def _test_fp16(self, gradient_as_bucket_view=False):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        gpus = gpus_for_rank(self.world_size)[self.rank]
        model = nn.Linear(1, 1, bias=False).cuda(gpus[0]).half()
        nn.init.constant_(model.weight, 1)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[gpus[0]],
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Input 2**15, so that the gradients will overflow with a
        # world_size of 2, unless we normalize the gradient by the
        # world_size before the reduction
        input = torch.tensor([[2 ** 15]]).cuda(gpus[0]).half()

        # Step model
        ddp_model.train()
        output = ddp_model(input)
        loss = output.sum()
        loss.backward()

        self.assertFalse(any(torch.isinf(p.grad).any() for p in ddp_model.parameters()))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16(self):
        self._test_fp16()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_grad_is_view(self):
        self._test_fp16(gradient_as_bucket_view=True)

    def _test_arbitrary_forward_return_value(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class ForwardReturnValueModule(nn.Module):
            def __init__(self):
                super(ForwardReturnValueModule, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x, fn):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # The first softmax does NOT include fc3 in its autograd graph
                # whereas the second softmax DOES. If we pass only the first
                # tensor we see in the output to the reducer, it marks the
                # gradient for fc3 as ready (because it doesn't show up). If
                # downstream uses of this return value choose to differentiate
                # against the second output tensor, it would still receive a
                # gradient and a callback for this tensor, resulting in a crash.
                return fn(
                    F.softmax(x, dim=1),
                    F.softmax(self.fc3(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            ForwardReturnValueModule().float().to(device_id),
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

        # Always run "backward" to ensure the reducer is called by autograd.
        # If we don't correctly capture the output tensors from the return value,
        # the reducer won't see a hook for the unused parameter, and throw an error.
        # The correct capture is what we're testing in this function.
        def test(box, unbox):
            output = model(input, fn=box)
            loss = criterion(unbox(output), target)
            loss.backward()

        # Test with identity return value
        test(
            box=lambda x, y: (x, y),
            unbox=lambda obj: obj[1],
        )

        # Test with list return value
        test(
            box=lambda x, y: ["foo", x, "bar", y],
            unbox=lambda obj: obj[3],
        )

        # Test with tuple return value
        test(
            box=lambda x, y: ("foo", x, "bar", y),
            unbox=lambda obj: obj[3],
        )

        # Test with dict return value
        test(
            box=lambda x, y: {"foo": "bar", "a": x, "b": y},
            unbox=lambda obj: obj["b"],
        )

        # Test with list with dict return value
        test(
            box=lambda x, y: ["foo", "bar", {"a": x, "b": y}],
            unbox=lambda obj: obj[2]["b"],
        )

        # Test with dict with list return value
        test(
            box=lambda x, y: {"foo": "bar", "list": [0, x, 1, y]},
            unbox=lambda obj: obj["list"][3],
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value(self):
        self._test_arbitrary_forward_return_value()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value_grad_is_view(self):
        self._test_arbitrary_forward_return_value(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_with_lazy_parameters(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        with self.assertRaisesRegex(
                RuntimeError, "Modules with uninitialized parameters"
        ):
            DistributedDataParallel(
                torch.nn.LazyLinear(10), process_group=process_group
            )

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
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_find_unused_parameters_kwarg_debug_detail(self):
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_find_unused_parameters_kwarg_debug_info(self):
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_find_unused_parameters_kwarg_debug_off(self):
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_detail(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_info(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_off(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

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

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward(self):
        self._test_multiple_outputs_multiple_backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward_grad_is_view(self):
        self._test_multiple_outputs_multiple_backward(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_no_grad(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class NoGradModule(nn.Module):
            def __init__(self):
                super(NoGradModule, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            NoGradModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        input = torch.rand([batch_size, 2], dtype=torch.float)

        def check_no_grads():
            for p in model.parameters():
                self.assertTrue(p.requires_grad)
                self.assertIsNone(p.grad)

        # After initialization, no parameter has their gradient set.
        check_no_grads()

        # Run `forward` function with torch.no_grad()
        with torch.no_grad():
            output = model(input)
            self.assertTrue(isinstance(output, torch.Tensor))

        # No parameter should have their gradient set.
        check_no_grads()

    def _test_accumulate_gradients_module(self, gradient_as_bucket_view=False):
        # This is NOT the recommended way to implement accumulating grads, but
        # we would like to make sure DDP does not mess up with the underlying
        # module.
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        global_batch_size = self.world_size

        model, ddp_model, input, target = self._prepare_single_device_module(
            process_group, devices, devices, global_batch_size, gradient_as_bucket_view
        )

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()

        # ensure accumulate grads works with no_grad
        with torch.no_grad():
            ddp_model.train()
            ddp_model.module(input)

        # Check two model parameters over 4 iterations.
        # Use 4 iterations because we alternate between reducing and
        # not reducing and want to make sure we switch both ways.
        for iteration in range(4):
            step_model(model, input, target)

            if iteration % 2 == 0:
                # Skip gradients sync without calling prepare_for_backward
                step_model(
                    ddp_model.module,
                    input[self.rank: (self.rank + 1)],
                    target[self.rank: (self.rank + 1)],
                )
                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    self.assertNotEqual(i.grad, j.grad)
            else:
                step_model(
                    ddp_model,
                    input[self.rank: (self.rank + 1)],
                    target[self.rank: (self.rank + 1)],
                )
                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                    self.assertEqualIgnoreType(i.grad, j.grad)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_accumulate_gradients_module(self):
        self._test_accumulate_gradients_module()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_accumulate_gradients_module_with_grad_is_view(self):
        self._test_accumulate_gradients_module(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_failure_recovery(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # need to create a separate file for the recovered FileStore, because
        # the original one will be deleted when destructing the first FileStore.
        recovery_filename = self.file_name + "_recovery"

        if self.rank == 0:
            # the file will be deleted by the recovered FileStore
            open(recovery_filename, "w").close()

        # not necessary to run barrier here, as DDP will synchronize

        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = TestModel().float().to(device_id)
        ddp = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

        del ddp
        del process_group
        del store  # this will delete self.file_name

        store = c10d.FileStore(recovery_filename, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        ddp = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )
        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_pass_default_pg(self):
        dist.init_process_group(
            "nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

        default_pg = c10d.distributed_c10d._get_default_group()
        dist.destroy_process_group(default_pg)
        self.assertFalse(dist.is_initialized())

    def _test_grad_layout(self, replica_devices, layer_devs, local_batch_size):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        global_batch_size = local_batch_size * self.world_size

        # Carry out some trials with small buckets and some with big buckets.
        bucketsizes = (0.000001, 25)
        # Tuples of lists.  Each list describes per-layer characteristics for one trial.
        layer_formats = (
            [torch.contiguous_format] * 4,
            [torch.channels_last] * 2 + [torch.contiguous_format] * 2,
            [torch.channels_last] * 4,
        )
        layer_dtypes = (
            [torch.float] * 4,
            [torch.float] * 2 + [torch.half] * 2,
            [torch.half] * 4,
        )

        input_dev = layer_devs[0] if isinstance(layer_devs, list) else layer_devs
        target_dev = layer_devs[-1] if isinstance(layer_devs, list) else layer_devs
        input = torch.randn(
            (global_batch_size, 8, 8, 8), device=input_dev, dtype=torch.float
        )
        target = torch.randn(
            (global_batch_size, 8, 4, 4), device=target_dev, dtype=torch.float
        )
        local_batch_start = self.rank * local_batch_size
        local_batch_end = (self.rank + 1) * local_batch_size

        # Reducer.cpp sneakily creates one "initial bucket" that ignores the "bucket_cap_mb"
        # argument.  The following makes sure the initial bucket also complies.
        @contextmanager
        def first_bucket_size(ddp_bucket_mb):
            old_DEFAULT_FIRST_BUCKET_BYTES = dist._DEFAULT_FIRST_BUCKET_BYTES
            dist._DEFAULT_FIRST_BUCKET_BYTES = int(ddp_bucket_mb * 1.0e6)
            try:
                yield
            finally:
                dist._DEFAULT_FIRST_BUCKET_BYTES = old_DEFAULT_FIRST_BUCKET_BYTES

        with torch.backends.cudnn.flags(
                enabled=True, deterministic=True, benchmark=False
        ):
            for formats, dtypes, bucketsize in product(
                    layer_formats, layer_dtypes, bucketsizes
            ):
                with first_bucket_size(bucketsize):
                    model_msg = (
                        "rank = {} formats = {} dtypes = {} bucketsize = {} ".format(
                            self.rank, formats, dtypes, bucketsize
                        )
                    )
                    try:
                        m = ConvNet(layer_devs, formats, dtypes)
                        m_ddp = DistributedDataParallel(
                            copy.deepcopy(m),
                            device_ids=replica_devices,
                            process_group=process_group,
                            bucket_cap_mb=bucketsize,
                        )
                        opt = torch.optim.SGD(m.parameters(), lr=0.1)
                        opt_ddp = torch.optim.SGD(m_ddp.parameters(), lr=0.1)
                        has_half = any(p.dtype is torch.half for p in m.parameters())
                        tol = 1.0e-3 if has_half else 1.0e-5
                    except BaseException:
                        # Prints case-specific debugging info to narrow down failing case.
                        print(
                            "Caught exception during model creation for " + model_msg,
                            flush=True,
                        )
                        raise
                    # 3 iters:  First iter creates grads, second iter retests after rebucketing,
                    # third iter tries zeroed grads.
                    for it in range(3):
                        iter_msg = "iter = {} ".format(it) + model_msg
                        named_msg = iter_msg
                        try:
                            F.mse_loss(m(input).float(), target).backward()
                            F.mse_loss(
                                m_ddp(input[local_batch_start:local_batch_end]).float(),
                                target[local_batch_start:local_batch_end],
                            ).backward()
                            for i, ((layer_name, m_child), m_ddp_child) in enumerate(
                                    zip(m.named_children(), m_ddp.module.children())
                            ):
                                named_msg = layer_name + ".weight" + " " + iter_msg
                                self.assertTrue(
                                    m_child.weight.grad.is_contiguous(
                                        memory_format=formats[i]
                                    ),
                                    named_msg,
                                )
                                self.assertTrue(
                                    m_ddp_child.weight.grad.is_contiguous(
                                        memory_format=formats[i]
                                    ),
                                    named_msg,
                                )
                                for j, ((param_name, p), p_ddp) in enumerate(
                                        zip(
                                            m_child.named_parameters(),
                                            m_ddp_child.parameters(),
                                        )
                                ):
                                    named_msg = (
                                        layer_name + "." + param_name + " " + iter_msg
                                    )
                                    self.assertEqual(
                                        p.grad, p_ddp.grad, rtol=tol, atol=tol
                                    )
                            opt.step()
                            opt_ddp.step()
                            if it == 0:
                                for p, p_ddp in zip(m.parameters(), m_ddp.parameters()):
                                    p.grad = None
                                    p_ddp.grad = None
                            else:
                                m.zero_grad()
                                m_ddp.zero_grad()
                        except BaseException:
                            # Makes sure we still get info if an error occurred somewhere other than the asserts.
                            print(
                                "Caught exception during iterations at " + named_msg,
                                flush=True,
                            )
                            raise

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm
    def test_grad_layout_1devicemodule_1replicaperprocess(self):
        dev0 = torch.device("cuda:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        # Tells DDP to use just one device.
        replica_devices = [dev0]
        # Tells _test_grad_layout to construct ConvNet with all layers on this process's first assigned device.
        layer_devs = dev0
        local_batch_size = 8
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_if_rocm
    def test_grad_layout_2devicemodule(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        dev0 = torch.device("cuda:" + str(int_devices[0]))
        dev1 = torch.device("cuda:" + str(int_devices[1]))
        # DDP's default behavior for a multi-device module is "don't replicate."
        replica_devices = None
        # Tells _test_grad_layout to constructs this process's ConvNet on 2 devices, with 2 layers on each device.
        layer_devs = [dev0] * 2 + [dev1] * 2
        local_batch_size = 8
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_param_layout_mismatch_error(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        dev0 = torch.device("cuda:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        layer_devs = dev0
        layer_formats = (
            [torch.contiguous_format] * 4
            if self.rank == 0
            else [torch.channels_last] * 4
        )
        layer_dtypes = [torch.float] * 4

        m = ConvNet(layer_devs, layer_formats, layer_dtypes)
        if self.rank == 0:
            m_ddp = DistributedDataParallel(
                m, device_ids=[dev0], process_group=process_group
            )
        else:
            with self.assertRaisesRegex(
                    RuntimeError,
                    ".* appears not to match strides of the same param in process 0",
            ):
                m_ddp = DistributedDataParallel(
                    m, device_ids=[dev0], process_group=process_group
                )

    def _gpu_model_with_ddp_comm_hook(
            self, process_group, hook=None, gradient_as_bucket_view=False, state=None, static_graph=False
    ):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        if static_graph:
            gpu_model._set_static_graph()

        # Register a DDP communication hook if any.
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_nccl(self):
        """
        This unit test verifies whether the Future object is passed properly using nccl backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # Get GPU model with simple_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # check whether the grads are equal to what simple_hook's then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    def _test_ddp_comm_hook_allreduce_hook_nccl(self, gradient_as_bucket_view=False, static_graph=False):
        """
        This unit test verifies whether a DDP communication hook that just calls
        allreduce gives the same result with the case of no hook registered.
        Without the then callback, the future_value in reducer is no longer
        a PyObject, and this unit test verifies future_value is properly checked.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allreduce_hook(state: object, bucket: dist.GradBucket) -> torch._C.Future:
            tensors = [bucket.get_tensor() / self.world_size]
            return process_group.allreduce(tensors).get_future()

        # Get GPU model with allreduce_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_hook, gradient_as_bucket_view, static_graph
        )

        # check whether the grads are equal to what DDP without hook would return.
        self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_default_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether default Python DDP communication hooks ALLREDUCE and FP16_COMPRESS
        can give the same result with the case of no hook registered.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # For these default DDP comm hooks, the only state is process group.
        state = process_group
        for hook in [default.allreduce_hook, default.fp16_compress_hook]:
            # Get GPU model with the hook registered.
            # The first arg 'process_group' is used for initializing the test environment,
            # so it cannot be replaced by 'state', although they have the same value.
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group, hook, gradient_as_bucket_view, state
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_fp16_compress_wrapper(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether wrapping the ALLREDUCE and POWER_SGD hooks with
        the FP16_WRAPPER can give the same result as when there is no hook registered.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        powerSGD_state = powerSGD.PowerSGDState(process_group=process_group)

        hook_args = [(powerSGD.powerSGD_hook, powerSGD_state), (default.allreduce_hook, process_group)]

        for hook, state in hook_args:
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group,
                default.fp16_compress_wrapper(hook),
                gradient_as_bucket_view,
                state
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_powerSGD_ddp_comm_hook_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether Python DDP communication hook POWER_SGD
        can give the same result with the case of no hook registered.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # Get GPU model with the hook registered.
        # Test the hook with different algorithmic configs.
        for use_error_feedback, warm_start in product([True, False], [True, False]):
            state = powerSGD.PowerSGDState(
                process_group=process_group,
                matrix_approximation_rank=1,
                use_error_feedback=use_error_feedback,
                warm_start=warm_start,
            )
            for hook in [powerSGD.powerSGD_hook, powerSGD.batched_powerSGD_hook]:
                gpu_model = self._gpu_model_with_ddp_comm_hook(
                    process_group, hook, gradient_as_bucket_view, state
                )

                # check whether the grads are equal to what DDP without hook would return.
                self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_builtin_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether built-in C++ DDP communication hooks ALLREDUCE and FP16_COMPRESS
        can give the same result with the case of no hook registered.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        for comm_hook_type in [
            dist.BuiltinCommHookType.ALLREDUCE,
            dist.BuiltinCommHookType.FP16_COMPRESS,
        ]:
            # Get GPU model with the built-in communication hook.
            gpu_model = self._gpu_model_with_builtin_ddp_comm_hook(
                process_group, comm_hook_type, gradient_as_bucket_view
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_nccl(self):
        self._test_default_ddp_comm_hooks_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_nccl(self):
        self._test_fp16_compress_wrapper()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl(self):
        self._test_builtin_ddp_comm_hooks_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl(self):
        self._test_powerSGD_ddp_comm_hook_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl_grad_is_view(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl_static_graph(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl(static_graph=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_nccl_is_view(self):
        self._test_default_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_is_view(self):
        self._test_fp16_compress_wrapper(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl_grad_is_view(self):
        self._test_builtin_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl_grad_is_view(self):
        self._test_powerSGD_ddp_comm_hook_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_with_then_hook_nccl(self):
        """
        This unit test verifies whether a DDP communication hook that calls allreduce and then
        multiplies the result by ten and divides by two gives the expected result.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allreduce_with_then_hook(
                state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future:
            tensors = [bucket.get_tensor() / self.world_size]
            fut = process_group.allreduce(tensors).get_future()

            def mult(fut):
                # Multiply the result by 10.
                return [10 * t for t in fut.value()]

            def div(fut):
                # Divide the result by 2.
                return [0.5 * t for t in fut.value()]

            return fut.then(mult).then(div)

        # Get GPU model with allreduce_with_then_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_with_then_hook
        )

        # check whether the grads are equal to what allreduce returns multuplied by 5.
        # without the comm_hook, result would be still 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 1.25 * torch.ones(2, 2))

    class AcceptsParam(torch.nn.Module):
        def __init__(self, p, factor):
            super().__init__()
            self.a = p
            self.f = factor

        def forward(self, input):
            return input + self.a * self.f

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_weight_sharing(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        size = 2048 * 2048
        dev = self.rank
        world = self.world_size

        p = torch.nn.Parameter(torch.randn(size, requires_grad=True))

        for try_set_to_none, use_bucket_view in product((False, True), (False, True)):
            m = torch.nn.Sequential(self.AcceptsParam(p, dev + 1),
                                    self.AcceptsParam(p, dev + 1)).cuda(dev)

            m = torch.nn.parallel.DistributedDataParallel(m,
                                                          bucket_cap_mb=1,
                                                          gradient_as_bucket_view=use_bucket_view,
                                                          device_ids=[dev],
                                                          process_group=process_group)

            for i in range(3):
                m.zero_grad(set_to_none=try_set_to_none)
                m(1).sum().backward()

                # Each param value is multiplied by "rank + 1" twice in forward, so the grad
                # values produced by a particular rank should be 2. * (rank + 1).
                # Summing these over ranks and dividing by world size gives the expected result:
                analytic = torch.full_like(p, 2. * (world * (world + 1.) / 2.) / world, device=dev)
                for name, p in m.named_parameters():
                    self.assertEqual(p.grad, analytic, "mismatch at " + name + ".grad for " +
                                     "set_to_none = {}, use_bucket_view = {}".format(try_set_to_none,
                                                                                     use_bucket_view))

    # A list of tests for ddp with activation checkpointing
    # when gradient_as_bucket_view=True, False.
    # Most of the tests are referred to
    # https://github.com/facebookresearch/fairscale/blob/master/tests/nn/pipe/test_checkpoint_ddp.py
    class CheckpointOnceModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(20, 20)
            self.l2 = nn.Linear(20, 20)

        def forward(self, inp):
            x = self.l1(inp)
            x = checkpoint(self.l2, x)
            return x

    class CheckpointTwiceModule(CheckpointOnceModule):
        def __init__(self):
            super().__init__()

        def forward(self, inp):
            x = self.l1(inp)
            x = checkpoint(self.l2, x)
            x = checkpoint(self.l2, x)
            return x

    def _prepare_dummy_data(self):
        ddp_bs = 16
        bs = ddp_bs * self.world_size
        input = torch.rand((bs, 20), device="cuda", requires_grad=True)
        target = torch.randn((bs, 20), device="cuda")
        offset = self.rank * ddp_bs
        ddp_input = input[offset: offset + ddp_bs]
        ddp_target = target[offset: offset + ddp_bs]
        return input, ddp_input, target, ddp_target

    def _train_model(self, model, input_var, target, loss, run_checkpoint=False):
        model.train()
        if run_checkpoint:
            output = checkpoint(model, input_var)
        else:
            output = model(input_var)
        l = loss(output, target)
        l.backward()

    def _test_ddp_checkpointing(
        self,
        input_model,
        process_group,
        use_bucket_view,
        find_unused_parameters=False,
        static_graph=False,
        run_checkpoint=False
    ):
        # to reprodce the same training results
        torch.cuda.set_device(self.rank)
        torch.manual_seed(31415)
        model = copy.deepcopy(input_model).cuda()
        ddp_model = copy.deepcopy(input_model).cuda()
        ddp_model = nn.parallel.DistributedDataParallel(
            ddp_model,
            bucket_cap_mb=1,
            gradient_as_bucket_view=use_bucket_view,
            device_ids=[self.rank],
            process_group=process_group,
            find_unused_parameters=find_unused_parameters
        )
        if static_graph:
            ddp_model._set_static_graph()
        self.assertEqual(ddp_model._get_ddp_logging_data().get("static_graph", 0), static_graph)
        input, ddp_input, target, ddp_target = self._prepare_dummy_data()
        loss = nn.MSELoss()
        for i in range(5):
            model.zero_grad(set_to_none=False)
            ddp_model.zero_grad(set_to_none=False)
            self._train_model(model, input, target, loss, run_checkpoint=run_checkpoint)
            self._train_model(ddp_model, ddp_input, ddp_target, loss, run_checkpoint=run_checkpoint)
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                self.assertTrue(i.grad is not None)
                self.assertTrue(j.grad is not None)
                self.assertEqual(i.grad, j.grad)

    # DDP works as expect when layer is checkpointed only once
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_once(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        for use_bucket_view, static_graph in product((False, True), (False, True)):
            self._test_ddp_checkpointing(self.CheckpointOnceModule(),
                                         process_group=process_group,
                                         use_bucket_view=use_bucket_view,
                                         static_graph=static_graph)

    # DDP will fail when there are unused_parameters in the model
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_unused_params(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        for use_bucket_view in (True, False):
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected to mark a variable ready only once.",
            ):
                model = self._test_ddp_checkpointing(self.CheckpointOnceModule(),
                                                     process_group=process_group,
                                                     use_bucket_view=use_bucket_view,
                                                     find_unused_parameters=True,
                                                     static_graph=False)
            # test passes when static_graph is true
            model = self._test_ddp_checkpointing(self.CheckpointOnceModule(),
                                                 process_group=process_group,
                                                 use_bucket_view=use_bucket_view,
                                                 find_unused_parameters=True,
                                                 static_graph=True)

    # DDP will fail when the same layer is checkponted twice
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_twice(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        for use_bucket_view in (True, False):
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected to mark a variable ready only once.",
            ):
                model = self._test_ddp_checkpointing(self.CheckpointTwiceModule(),
                                                     process_group=process_group,
                                                     use_bucket_view=use_bucket_view,
                                                     static_graph=False)
            model = self._test_ddp_checkpointing(self.CheckpointTwiceModule(),
                                                 process_group=process_group,
                                                 use_bucket_view=use_bucket_view,
                                                 static_graph=True)

    # DDP works as expected if there is weight sharing among layers
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_weight_sharing(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        torch.cuda.set_device(self.rank)
        for use_bucket_view, static_graph in product((False, True), (False, True)):
            torch.manual_seed(31415)
            l1 = nn.Linear(20, 20)
            l2 = nn.Linear(20, 20)
            l1.weight = l2.weight
            model = nn.Sequential(l1, l2)
            self._test_ddp_checkpointing(model,
                                         process_group=process_group,
                                         use_bucket_view=use_bucket_view,
                                         static_graph=static_graph,
                                         run_checkpoint=True)


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
