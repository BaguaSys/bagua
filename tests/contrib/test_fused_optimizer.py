import bagua.torch_api as bagua
import torch
import torch.optim as optim
import unittest
import os
from tests.internal.common_utils import find_free_port
from tests import skip_if_cuda_available, skip_if_cuda_not_available

import logging

logging.getLogger().setLevel(logging.INFO)


def construct_model_and_optimizer(opt, flag_param, device):
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

    return model, optimizer


def train_model(model, optimizer, device, num_epochs):
    input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], device=device).reshape(3, 2)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        output = model(input)
        loss = output.sum()
        loss.backward()

        optimizer.step()


def train_model_fused(model, optimizer, device, num_epochs):
    input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], device=device).reshape(3, 2)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input)
        loss = output.sum()
        loss.backward()

        if epoch % 2 == 0:
            optimizer.fuse_step()
        else:
            optimizer.step()


def run(opt, flag_param, device, num_epochs):
    model, optimizer = construct_model_and_optimizer(opt, flag_param, device)

    train_model(model, optimizer, device, num_epochs=num_epochs)
    return model.parameters()


def run_fused(opt, flag_param, device, num_epochs):
    model, optimizer = construct_model_and_optimizer(opt, flag_param, device)
    optimizer = bagua.contrib.fuse_optimizer(optimizer, do_flatten=True)

    train_model_fused(model, optimizer, device, num_epochs=num_epochs)
    return model.parameters()


def run_fused_bagua(opt, flag_param, device, num_epochs):
    model, optimizer = construct_model_and_optimizer(opt, flag_param, device)
    optimizer = bagua.contrib.fuse_optimizer(optimizer, do_flatten=True)

    model.with_bagua(
        [optimizer],
        bagua.algorithms.gradient_allreduce.GradientAllReduceAlgorithm(),
        do_flatten=False,
    )

    train_model_fused(model, optimizer, device, num_epochs=num_epochs)
    return model.parameters()


def run_fused_bagua_v2(opt, flag_param, device, num_epochs):
    model, optimizer = construct_model_and_optimizer(opt, flag_param, device)
    optimizer = bagua.contrib.fuse_optimizer(optimizer, do_flatten=True)

    model.with_bagua(
        [optimizer],
        bagua.algorithms.gradient_allreduce.GradientAllReduceAlgorithm(),
        do_flatten=True,
    )

    train_model_fused(model, optimizer, device, num_epochs)
    return model.parameters()


def run_fused_bagua_with_states(opt, flag_param, device, num_epochs):
    model, optimizer = construct_model_and_optimizer(opt, flag_param, device)
    train_model(model, optimizer, device, 1)

    optimizer = bagua.contrib.fuse_optimizer(optimizer, do_flatten=True)
    model.with_bagua(
        [optimizer],
        bagua.algorithms.gradient_allreduce.GradientAllReduceAlgorithm(),
        do_flatten=False,
    )

    train_model_fused(model, optimizer, device, num_epochs - 1)
    return model.parameters()


class TestFusedOptimizer(unittest.TestCase):
    def run_all_optimizers_once(self, fn1, fn2, device, num_epochs):
        optimizer_list = [
            optim.SGD,
            optim.SGD,
            optim.Adam,
            optim.Adam,
            optim.Adam,
            optim.Adam,
            optim.AdamW,
            optim.AdamW,
            optim.AdamW,
            optim.AdamW,
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
            dict(lr=0.2, momentum=1, dampening=0, weight_decay=1, nesterov=True),  # SGD
            dict(
                lr=0.2, momentum=1, dampening=0.5, weight_decay=1, nesterov=False
            ),  # SGD
            dict(weight_decay=1.0, amsgrad=True),  # Adam
            dict(weight_decay=1.0, amsgrad=False),  # Adam
            dict(weight_decay=0.0, amsgrad=True),  # Adam
            dict(weight_decay=0.0, amsgrad=False),  # Adam
            dict(weight_decay=1.0, amsgrad=True),  # AdamW
            dict(weight_decay=1.0, amsgrad=False),  # AdamW
            dict(weight_decay=0.0, amsgrad=True),  # AdamW
            dict(weight_decay=0.0, amsgrad=False),  # AdamW
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

        count = 0
        for opt, flag_param in zip(optimizer_list, flag_params):
            res1 = fn1(opt, flag_param, device=device, num_epochs=num_epochs)
            res2 = fn2(opt, flag_param, device=device, num_epochs=num_epochs)

            for p1, p2 in zip(res1, res2):
                self.assertTrue(torch.equal(p1, p2))

            count += 1
            logging.info(f"Tests Passed [{count}/{len(optimizer_list)}]")

    @skip_if_cuda_available()
    def test_fused_optimizer(self):
        self.run_all_optimizers_once(
            fn1=run, fn2=run_fused, device="cpu", num_epochs=1001
        )

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

        self.run_all_optimizers_once(
            fn1=run, fn2=run_fused_bagua, device="cuda:0", num_epochs=1001
        )
        self.run_all_optimizers_once(
            fn1=run, fn2=run_fused_bagua_v2, device="cuda:0", num_epochs=1001
        )
        self.run_all_optimizers_once(
            fn1=run, fn2=run_fused_bagua_with_states, device="cuda:0", num_epochs=1001
        )

    @skip_if_cuda_available()
    def test_calculate_mutual_groups(self):
        from bagua.torch_api.contrib.fuse.optimizer import calculate_mutual_groups

        tensor = torch.rand(100)

        tensor_pieces = []
        for i in range(10):
            tensor_pieces.append(tensor[i * 10 : (i + 1) * 10])

        g1 = [
            tensor_pieces[3],
            tensor_pieces[1],
            tensor_pieces[2],
            tensor_pieces[0],
            tensor_pieces[8],
            tensor_pieces[9],
            torch.rand(10),
        ]
        g2 = [torch.rand(10) for i in range(len(g1))]
        g3 = [
            tensor_pieces[3],
            tensor_pieces[1],
            tensor_pieces[2],
            tensor_pieces[0],
            torch.rand(10),
            torch.rand(10),
            torch.rand(10),
        ]
        g4 = [
            torch.rand(10),
            tensor_pieces[1],
            tensor_pieces[2],
            tensor_pieces[0],
            tensor_pieces[8],
            tensor_pieces[9],
            torch.rand(10),
        ]
        g5 = [
            tensor_pieces[3],
            tensor_pieces[1],
            tensor_pieces[2],
            tensor_pieces[0],
            tensor_pieces[8],
            tensor_pieces[9],
            torch.rand(10),
        ]

        ret = calculate_mutual_groups([g1, g2])
        self.assertTrue(ret == [])

        ret = calculate_mutual_groups([g1, g3])
        self.assertTrue(ret == [[3, 1, 2, 0]])

        ret = calculate_mutual_groups([g1, g4])
        self.assertTrue(ret == [[4, 5]])

        ret = calculate_mutual_groups([g1, g5])
        self.assertTrue(ret == [[3, 1, 2, 0], [4, 5]])


if __name__ == "__main__":
    unittest.main()
