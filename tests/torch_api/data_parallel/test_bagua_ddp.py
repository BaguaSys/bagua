import os
import random
import sys
import unittest

import torch
import torch.distributed as c10d

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.nn.functional as F
from torch import nn
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TSAN,
)
from . import test_c10d_common

import bagua.torch_api as bagua
from tests.internal.common_utils import find_free_port
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
        os.environ.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(find_free_port(8000, 8100)),
                "BAGUA_SERVICE_PORT": str(find_free_port(9000, 9100)),
            }
        )
        self._spawn_processes()

    def _test_find_unused_parameters_kwarg(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        torch.cuda.set_device(self.rank)
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

        device_id = self.rank
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float).to(device_id)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        ddp_model = None

        def test_find_unused_parameters(
            find_unused_parameters
        ):
            model = DistributedDataParallel(
                FindUnusedParametersModule().float().to(device_id),
                device_ids=[device_id],
                process_group=process_group,
                find_unused_parameters=find_unused_parameters,
            )
            nonlocal ddp_model
            ddp_model = model

            output, fc3 = model(input)
            # output = fc3(output)
            loss = criterion(output, target)
            loss.backward()

        test_find_unused_parameters(find_unused_parameters=True)
        bagua_build_params = [name for name,
                              _ in ddp_model.inner.bagua_build_params()]
        self.assertEqual(set(bagua_build_params),
                         set(['fc1.weight', 'fc2.weight']))

        test_find_unused_parameters(find_unused_parameters=False)
        bagua_build_params = [name for name,
                              _ in ddp_model.inner.bagua_build_params()]
        self.assertEqual(set(bagua_build_params), set(
            ['fc1.weight', 'fc2.weight', 'fc3.weight']))

    @skip_if_lt_x_gpu(2)
    def test_find_unused_parameters_kwarg_debug_detail(self):
        os.environ.update(
            {
                "WORLD_SIZE": str(self.world_size),
                "LOCAL_WORLD_SIZE": str(self.world_size),
                "RANK": str(self.rank),
                "LOCAL_RANK": str(self.rank),
            }
        )

        self._test_find_unused_parameters_kwarg()


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
