import bagua.torch_api as bagua
import torch
import torch.optim as optim
import unittest
import os
from tests.internal.common_utils import find_free_port
from tests import skip_if_cuda_available, skip_if_cuda_not_available


def run_step(opt, flag_param, fuse, wrap, device):
    weight = torch.tensor(
        [[-0.2109, -0.4976], [-0.1413, -0.3420], [-0.2524, 0.6976]],
        requires_grad=True,
    )
    bias = torch.tensor(
        [-0.1085, -0.2979, 0.6892],
        requires_grad=True,
    )
    weight2 = torch.tensor(
        [[-0.0508, -0.3941, -0.2843]],
        requires_grad=True,
    )
    bias2 = torch.tensor([-0.0711], requires_grad=True)

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(3, 1),
        torch.nn.Sigmoid(),
    )

    pretrained_dict = model.state_dict()
    pretrained_dict["0.weight"] = weight
    pretrained_dict["0.bias"] = bias
    pretrained_dict["2.weight"] = weight2
    pretrained_dict["2.bias"] = bias2
    model.load_state_dict(pretrained_dict)

    model = model.to(device)
    optimizer = opt(model.parameters(), **flag_param)

    if fuse:
        bagua.contrib.FusedOptimizer(optimizer, do_flatten=not wrap)

    if wrap:
        model.with_bagua(
            [optimizer],
            bagua.algorithms.gradient_allreduce.GradientAllReduceAlgorithm(),
        )

    input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], device=device).reshape(3, 2)

    for _ in range(1001):
        optimizer.zero_grad()
        output = model(input)
        loss = output.sum()
        loss.backward()

        optimizer.step()

    return model.parameters()


class TestFusedOptimizer(unittest.TestCase):
    def run_all_optimizers_once(self, wrap, device):
        optimizer_list = [
            optim.Adam,
            optim.Adam,
            optim.Adam,
            optim.Adam,
            optim.AdamW,
            optim.AdamW,
            optim.AdamW,
            optim.AdamW,
            optim.SGD,
            optim.SGD,
            optim.RMSprop,
            optim.RMSprop,
            optim.RMSprop,
            optim.RMSprop,
            optim.Rprop,
            optim.ASGD,
            optim.ASGD,
            optim.Adamax,
            optim.Adamax,
            optim.Adadelta,
            optim.Adadelta,
        ]

        flag_params = [
            dict(weight_decay=1.0, amsgrad=True),  # Adam
            dict(weight_decay=1.0, amsgrad=False),  # Adam
            dict(weight_decay=0.0, amsgrad=True),  # Adam
            dict(weight_decay=0.0, amsgrad=False),  # Adam
            dict(weight_decay=1.0, amsgrad=True),  # AdamW
            dict(weight_decay=1.0, amsgrad=False),  # AdamW
            dict(weight_decay=0.0, amsgrad=True),  # AdamW
            dict(weight_decay=0.0, amsgrad=False),  # AdamW
            dict(lr=0.2, momentum=1, dampening=0, weight_decay=1, nesterov=True),  # SGD
            dict(
                lr=0.2, momentum=1, dampening=0.5, weight_decay=1, nesterov=False
            ),  # SGD
            dict(weight_decay=1, momentum=1, centered=True),  # RMSprop
            dict(weight_decay=1, momentum=0, centered=True),  # RMSprop
            dict(weight_decay=1, momentum=1, centered=False),  # RMSprop
            dict(weight_decay=0, momentum=1, centered=False),  # RMSprop
            dict(lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)),  # Rprop
            dict(weight_decay=0),  # ASGD
            dict(weight_decay=1),  # ASGD
            dict(weight_decay=0),  # Adamax
            dict(weight_decay=1),  # Adamax
            dict(weight_decay=0),  # Adadelta
            dict(weight_decay=1),  # Adadelta
        ]

        for opt, flag_param in zip(optimizer_list, flag_params):
            res1 = run_step(opt, flag_param, fuse=True, wrap=wrap, device=device)
            res2 = run_step(opt, flag_param, fuse=False, wrap=wrap, device=device)

            for p1, p2 in zip(res1, res2):
                self.assertTrue(torch.equal(p1, p2))

    @skip_if_cuda_available()
    def test_fused_optimizer(self):
        self.run_all_optimizers_once(device="cpu", wrap=False)

    @skip_if_cuda_not_available()
    def test_fused_optimizer_with_bagua_wrapper(self):
        # init env
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port(8000, 8100))
        os.environ["BAGUA_SERVICE_PORT"] = str(find_free_port(9000, 9100))

        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"

        # init bagua distributed process group
        torch.cuda.set_device(0)
        bagua.init_process_group()

        self.run_all_optimizers_once(device="cuda:0", wrap=True)


if __name__ == "__main__":
    unittest.main()
