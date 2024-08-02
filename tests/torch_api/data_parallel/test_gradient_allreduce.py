import logging
import os
import pickle
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import bagua.torch_api as bagua

from bagua.torch_api.data_parallel import DistributedDataParallel as DDP
from tests.internal.multi_process_v2 import MultiProcessTestCase, skip_if_lt_x_gpu

logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=True)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def run_model(hierarchical):
    # construct model and optimizer, etc.
    model = Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    def run_epochs(num_epochs):
        for _ in range(num_epochs):
            data = torch.randn(4, 2).cuda()
            target = torch.randn(4, 4).cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)

            loss.backward()
            optimizer.step()

    run_epochs(1)

    # wrap model
    model = DDP(
        model,
        optimizers=[optimizer],
        algorithm=bagua.algorithms.gradient_allreduce.GradientAllReduceAlgorithm(
            hierarchical=hierarchical,
        ),
    )

    run_epochs(10)

    flattened_weight = bagua.utils.flatten([param.data for param in model.parameters()])
    weight_norm = flattened_weight.norm().item()
    return weight_norm


class TestGradientAllReduce(MultiProcessTestCase):
    def setUp(self):
        super(TestGradientAllReduce, self).setUp()
        self._spawn_processes()

    def tearDown(self):
        super(TestGradientAllReduce, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _check_result(self, test_id=None):
        result = None
        for i, process in enumerate(self.processes):
            _, msg = self.pid_to_pipe[process.pid].recv()
            weight_norm = pickle.loads(msg)

            logger.info("process {} result: {}".format(i, weight_norm))
            if result is None:
                result = weight_norm
            else:
                assert result == weight_norm

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_algorithm(self):
        # set deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.rank)

        self._init_bagua_distributed()
        return run_model(hierarchical=False)

    @skip_if_lt_x_gpu(4)
    def test_algorithm_hierarchical(self):
        # set deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.rank)

        self._init_bagua_distributed()
        return run_model(hierarchical=True)


if __name__ == "__main__":
    unittest.main()
