import os
import unittest
import multiprocessing
import itertools
import inspect
from multiprocessing import Manager
import logging
import bagua.torch_api as bagua
from tests.internal.common_utils import find_free_port
from tests import skip_if_cuda_not_available
import torch
import torch.nn as nn
import torch.nn.functional as F
from bagua.torch_api.data_parallel import DistributedDataParallel as DDP


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


def _init_bagua_env(rank, env):
    # Set deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(rank)
    # Initialize subprocess env
    os.environ["WORLD_SIZE"] = env["WORLD_SIZE"]
    os.environ["LOCAL_WORLD_SIZE"] = env["LOCAL_WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = env["MASTER_ADDR"]
    os.environ["MASTER_PORT"] = env["MASTER_PORT"]
    os.environ["BAGUA_SERVICE_PORT"] = env["BAGUA_SERVICE_PORT"]

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # Init bagua distributed process group
    torch.cuda.set_device(rank)
    bagua.init_process_group()


def create_model_and_optimizer(opt_class, opt_param):
    # C_in, C_out = 3, 10
    model = Net().cuda()
    hyper_param = {
        k: v
        for k, v in opt_param.items()
        if k in inspect.getargspec(opt_class.__init__).args
    }
    optimizer = opt_class(model.parameters(), **hyper_param)
    return model, optimizer


def get_optimizer_param_values(optimizer):
    results = []
    state_dict = optimizer.state_dict()
    for group in state_dict["param_groups"]:
        for param_id in group["params"]:
            if param_id not in state_dict["state"]:
                continue
            params = sorted(state_dict["state"][param_id].items())
            for k, v in params:
                results.append(
                    (k, v.clone().detach().cpu().numpy() if torch.is_tensor(v) else v)
                )
    return results


def run_bagua_broad(rank, nprocs, bagua_params, envs, opt_class, opt_hyper_param):
    _init_bagua_env(rank, envs)

    model, bagua_optimizer = create_model_and_optimizer(
        opt_class, opt_hyper_param
    )

    for epoch in range(5):
        logging.debug("Training epoch {}".format(epoch))
        for _ in range(10):
            data = torch.randn(4, 2).cuda()
            target = torch.randn(4, 4).cuda()

            bagua_optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, target)

            loss.backward()
            bagua_optimizer.step()

    from bagua.torch_api.algorithms import gradient_allreduce

    algorithm = gradient_allreduce.GradientAllReduceAlgorithm(hierarchical=True)
    ddp_model = DDP(model, optimizers=[bagua_optimizer], algorithm=algorithm)

    model_params = [
        (k, v.clone().detach().cpu().numpy())
        for k, v in sorted(ddp_model.state_dict().items())
    ]
    optimizer_params = get_optimizer_param_values(bagua_optimizer)

    # Put "model_params" in dimension 1, while "optimizer_params" in dimension 2.
    bagua_params[rank][0].extend(model_params)
    bagua_params[rank][1].extend(optimizer_params)


class Test_Broadcast_Module(unittest.TestCase):
    @skip_if_cuda_not_available()
    def test_broadcast_module(self):
        nprocs = torch.cuda.device_count()

        optimizers = [
            (optim_class.__name__, optim_class)
            for optim_class in [torch.optim.SGD, torch.optim.Adam, torch.optim.Rprop]
        ]

        optimizer_hyper_param = [
            dict(lr=0.2, momentum=0.9, weight_decay=0.1, centered=True),
            dict(lr=0.2),
        ]

        for (opt_name, opt_class), opt_hyper_param in itertools.product(
            optimizers, optimizer_hyper_param
        ):
            env = {
                "WORLD_SIZE": str(nprocs),
                "LOCAL_WORLD_SIZE": str(nprocs),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(find_free_port(8000, 8100)),
                "BAGUA_SERVICE_PORT": str(find_free_port(9000, 9100)),
            }
            with Manager() as manager:
                # For each rank, set a two dimensional list. One is used to save model_params,
                # while the second save optimizer_params.
                bagua_params = manager.list(
                    [[manager.list() for _ in range(2)] for _ in range(nprocs)]
                )
                mp = multiprocessing.get_context("spawn")
                processes = []
                for i in range(nprocs):
                    p = mp.Process(
                        target=run_bagua_broad,
                        args=(
                            i,
                            nprocs,
                            bagua_params,
                            env,
                            opt_class,
                            opt_hyper_param,
                        ),
                    )
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join(timeout=60)
                    self.assertTrue(p.exitcode == 0)
                for rank in range(0, nprocs):
                    # Both "model_params" and "optimizer_params" are saved in (name, tensor/scalar) form,
                    # so we need to assert the two dimensional separately.
                    # This is compare the "model_params".
                    for i in range(len(bagua_params[0][0])):
                        # assert name
                        self.assertEqual(
                            bagua_params[0][0][i][0],
                            bagua_params[rank][0][i][0],
                        )
                        # assert tensor
                        self.assertTrue(
                            torch.equal(
                                torch.tensor(
                                    bagua_params[0][0][i][1], dtype=torch.float
                                ),
                                torch.tensor(
                                    bagua_params[rank][0][i][1], dtype=torch.float
                                ),
                            )
                        )

                    if len(bagua_params[0][1]) != 0:
                        for j in range(len(bagua_params[0][1])):
                            # assert name
                            self.assertEqual(
                                bagua_params[0][1][j][0],
                                bagua_params[rank][1][j][0],
                            )
                            # assert tensor/scalar
                            if (
                                bagua_params[0][1][j][1] is None
                            ):  # this is for "torch.optim.sgd.SGD" and dict(lr=0.2)
                                continue
                            else:
                                self.assertTrue(
                                    torch.equal(
                                        torch.tensor(
                                            bagua_params[0][1][j][1], dtype=torch.float
                                        ),
                                        torch.tensor(
                                            bagua_params[rank][1][j][1],
                                            dtype=torch.float,
                                        ),
                                    )
                                )


if __name__ == "__main__":
    unittest.main()
