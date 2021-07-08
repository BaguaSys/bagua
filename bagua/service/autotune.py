# TODO: @shjwudp merge with service module
import copy
import collections
import logging
import os
import re
import skopt
import time
from typing import Tuple, Optional
from pssh.clients import ParallelSSHClient
from pssh.exceptions import Timeout


class IntParam:
    def __init__(self, val: int, space_dimension: Tuple[int, int]):
        self.val: int = val
        self.space_dimension: Tuple[int, int] = space_dimension

    def __str__(self):
        return str(self.__dict__)


class FloatParam:
    def __init__(self, val: float, space_dimension: Tuple[float, float]):
        self.val: float = val
        self.space_dimension: Tuple[float, float] = space_dimension

    def __str__(self):
        return str(self.__dict__)


class BoolParam:
    def __init__(self, val: bool):
        self.val: bool = val
        self.space_dimension: Tuple[int, int] = (0, 1)

    def __str__(self):
        return str(self.__dict__)


class BayesianOptimizer:
    """
    Simple package of beyasian optimizer
    """

    def __init__(
        self,
        param_declaration: dict,
        n_initial_points: int = 20,
        initial_point_generator: str = "halton",
        random_state: Optional[int] = 0,
    ):
        self.param_declaration = collections.OrderedDict(param_declaration)
        search_space = [
            declar.space_dimension for _, declar in self.param_declaration.items()
        ]

        self.bayesian_optimizer = skopt.Optimizer(
            dimensions=search_space,
            n_initial_points=n_initial_points,
            initial_point_generator=initial_point_generator,
            n_jobs=-1,
            random_state=random_state,
        )

    def tell(self, param_dict: dict, score: float) -> None:
        param_v = [
            float(param_dict[name])
            for name, _ in self.param_declaration.items()  # noqa: E501
        ]
        try:
            self.bayesian_optimizer.tell(param_v, -score)
        except ValueError as err:
            logging.warning(
                "Maybe sklearn's division by 0 bug, skip it. err={}, param_v={}".format(
                    err, param_v
                )
            )

    def ask(self) -> dict:
        param_v = self.bayesian_optimizer.ask()
        param_dict = {}
        for i, (name, _) in enumerate(self.param_declaration.items()):
            param_dict[name] = param_v[i]

        return param_dict


def sysperf(host_list, nproc_per_node, ssh_port, env: dict = {}):
    assert len(host_list) != 0, "Invalid host_list={}".format(host_list)

    if "PATH" not in env:
        env["PATH"] = os.environ["PATH"]

    pretreat_cmd = [
        "shopt -s huponexit &&",
    ]
    for k, v in env.items():
        pretreat_cmd.append(
            "{key}={value} &&".format(
                key=k,
                value=v,
            )
        )

    master_addr = host_list[0]
    host_args = []
    for i, _ in enumerate(host_list):
        host_args.append(
            {
                "cmd": " ".join(
                    pretreat_cmd
                    + [
                        "python -m bagua.distributed.launch",
                        "--nproc_per_node={}".format(nproc_per_node),
                        "--nnodes={} --node_rank={}".format(len(host_list), i),
                        '--master_addr="{}"'.format(master_addr),
                        "--master_port={}".format(8124),
                        "$(which bagua_sys_perf) --model vgg16",
                    ]
                ),
            }
        )

    client = ParallelSSHClient(host_list, port=ssh_port)
    output = client.run_command(
        "%(cmd)s",
        host_args=host_args,
        shell="bash -xc",
        use_pty=True,  # The key configuration of process safe exit
        read_timeout=60,
    )
    speed_pattern = re.compile(
        r"Total img/sec on (\d+) (\S+)\(s\): (\d*\.\d+|\d+) \+-(\d*\.\d+|\d+)"
    )
    host_out = output[0]
    m = None

    st = time.time()
    try:
        for line in host_out.stdout:
            print(line, flush=True)
            m = speed_pattern.search(line)
            if m:
                break
    except Timeout:
        print("Timeout 1, spend={}".format(time.time() - st))
        pass

    if m is None:
        return (None, None, 0.0, None)

    assert m, "no speed pattern, host_out.exit_code={}, host_out.stderr={}".format(
        host_out.exit_code, list(host_out.stderr)
    )

    ngpus = int(m.groups()[0])
    device = m.groups()[1]
    speed = float(m.groups()[2])
    speed_std = float(m.groups()[3])

    return (ngpus, device, speed, speed_std)


def autotune_system_hyperparameters(host_list, nproc_per_node, ssh_port):
    def _sysperf(env={}):
        result = sysperf(host_list, nproc_per_node, ssh_port, env=env)
        print(result)
        return result

    optim = BayesianOptimizer(
        {
            "NCCL_MIN_NCHANNELS": IntParam(
                val=0,  # 0 means no set
                space_dimension=(
                    0,
                    12,
                ),
            ),
            "NCCL_SOCKET_NTHREADS": IntParam(
                val=0,  # 0 means no set
                space_dimension=(
                    0,
                    8,
                ),
            ),
            "NCCL_NSOCKS_PERTHREAD": IntParam(
                val=0,  # 0 means no set
                space_dimension=(
                    0,
                    8,
                ),
            ),
            "nccl_buffsize_2p": IntParam(
                val=0,  # power of 2, 0 means no set
                space_dimension=(
                    0,
                    26,
                ),
            ),
        }
    )

    param_dict = {
        "NCCL_MIN_NCHANNELS": 0,
        "NCCL_SOCKET_NTHREADS": 0,
        "NCCL_NSOCKS_PERTHREAD": 0,
        "nccl_buffsize_2p": 0,
    }

    result_list = []
    for i in range(100):
        env_vars = copy.deepcopy(param_dict)
        for k, v in list(env_vars.items()):
            if v == 0:
                del env_vars[k]

        if "nccl_buffsize_2p" in env_vars:
            env_vars["NCCL_BUFFSIZE"] = 2 ** env_vars["nccl_buffsize_2p"]

        (_, _, speed, speed_std) = _sysperf(env=env_vars)
        result_list.append([copy.deepcopy(env_vars), speed, speed_std])
        optim.tell(param_dict, speed)
        param_dict = optim.ask()

    result_list = sorted(result_list, key=lambda x: -x[1])
    print(result_list)

    result_reduct = {}
    for (setting, speed, _) in result_list:
        key = tuple(sorted(setting.items()))
        if key not in result_reduct:
            result_reduct[key] = []
        result_reduct[key].append(speed)
    result_reduct = [
        [setting, sum(speed_list) / len(speed_list)]
        for (setting, speed_list) in result_reduct.items()
    ]
    result_reduct = sorted(result_reduct, key=lambda item: -item[1])
    print(result_reduct)

    return result_reduct[0][0]
