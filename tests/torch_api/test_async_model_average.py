import logging
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import bagua.torch_api as bagua

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


def create_model_and_optimizer(warmup_steps):
    # construct model and optimizer
    model = Net().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # wrap model
    algorithm = bagua.algorithms.async_model_average.AsyncModelAverageAlgorithm(
        sync_interval_ms=20,
        warmup_steps=warmup_steps,
    )
    model = model.with_bagua([optimizer], algorithm)

    return model, optimizer


def train_epoch(epoch, model, optimizer):
    logger.debug("Training epoch {}".format(epoch))
    for _ in range(10):
        data = torch.randn(4, 2).cuda()
        target = torch.randn(4, 4).cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)

        loss.backward()
        optimizer.step()


class TestAsyncModelAverage(MultiProcessTestCase):
    def setUp(self):
        super(TestAsyncModelAverage, self).setUp()
        self._spawn_processes()

    def tearDown(self):
        super(TestAsyncModelAverage, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_algorithm(self):
        self._init_bagua_distributed()
        model, optimizer = create_model_and_optimizer(warmup_steps=0)

        for epoch in range(100):
            train_epoch(epoch, model, optimizer)
        model.bagua_algorithm.abort(model)

    @skip_if_lt_x_gpu(4)
    def test_multiple_aborts(self):
        self._init_bagua_distributed()
        model, optimizer = create_model_and_optimizer(warmup_steps=10)

        for i in range(2):
            model.bagua_algorithm.resume(model)
            model.bagua_algorithm.abort(model)
            model.bagua_algorithm.resume(model)
            for epoch in range(100):
                train_epoch(i * 100 + epoch, model, optimizer)

            model.bagua_algorithm.abort(model)
            model.bagua_algorithm.abort(model)


if __name__ == "__main__":
    unittest.main()
