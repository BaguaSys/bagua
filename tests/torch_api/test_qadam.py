import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from bagua.torch_api.algorithms.q_adam import QAdamOptimizer
from tests import skip_if_cuda_available


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


def run_step(opt_cls, opt_flags, seed):
    torch.manual_seed(seed)
    model = Net()
    optimizer = opt_cls(model.parameters(), **opt_flags)

    for step in range(1000):
        data = torch.randn(4, 2)
        target = torch.randn(4, 4)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)

        loss.backward()
        optimizer.step()

    return loss


class TestQAdam(unittest.TestCase):
    @skip_if_cuda_available()
    def test_qadam_optimizer(self):
        loss1 = run_step(torch.optim.Adam, {"lr": 0.001, "weight_decay": 0.1}, seed=13)
        loss2 = run_step(
            QAdamOptimizer,
            {"lr": 0.001, "weight_decay": 0.1, "warmup_steps": 2000},
            seed=13,
        )
        self.assertEqual(loss1.item(), loss2.item())
