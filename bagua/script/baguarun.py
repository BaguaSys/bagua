#!/usr/bin/env python

"""
Multi-node launch script. execute bagua.distributed.launch on the specified host list.
The following is an example of running bagua program with two nodes:
```bash
# Started by baguarun
Launch bagua on two nodes *(192.168.1.1, 192.168.1.2)*
    >>> baguarun --host_list 192.168.1.1,192.168.1.2 --ssh_port ${COMM_SSH_PORT}
                --nproc_per_node=NUM_GPUS_YOU_HAVE --master_port=1234
                YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of your training script)
# The same operation do with bagua.distributed.launch
Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*
::
    >>> python -m bagua.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)
Node 2 *(IP: 192.168.1.2):
::
    >>> python -m bagua.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)
```
"""

import argparse
import os
import gevent
from pssh.clients import ParallelSSHClient
from pssh.config import HostConfig
from pssh.utils import enable_host_logger


def pssh_bagua_launch(
    args,
    script_cmd: str,
    env: dict = {},
):
    host_list = [host for (host, _) in args.host_list]
    nproc_per_node = args.nproc_per_node
    assert len(host_list) != 0, "Invalid host_list={}".format(host_list)

    if "PATH" not in env:
        env["PATH"] = os.environ["PATH"]
    if "LD_LIBRARY_PATH" not in env:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]

    pretreat_cmd = [
        "shopt -s huponexit; cd {};".format(os.getcwd()),
    ]
    for k, v in env.items():
        pretreat_cmd.append(
            "export {key}={value} &&".format(
                key=k,
                value=v,
            )
        )

    bypass_args = []
    if args.master_port:
        bypass_args.append("--master_port={}".format(args.master_port))
    if args.bagua_service_port:
        bypass_args.append(
            "--bagua_service_port={}".format(args.bagua_service_port)
        )  # noqa: E501
    if args.no_python:
        bypass_args.append("--no_python")
    if args.enable_bagua_net:
        bypass_args.append("--enable_bagua_net")

    host_config = [HostConfig(port=port) for (_, port) in args.host_list]

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
                    ]
                    + bypass_args
                    + [script_cmd]
                ),
            }
        )

    enable_host_logger()
    client = ParallelSSHClient(
        host_list, host_config=host_config, pool_size=10 + len(host_list)
    )
    output = client.run_command(
        "%(cmd)s",
        host_args=host_args,
        shell="bash -xc",
        use_pty=True,  # The key configuration of process safe exit
    )

    def do_output(host_out):
        for line in host_out.stdout:
            pass

    cmds = [client.pool.spawn(do_output, host_out) for host_out in output]
    gevent.joinall(cmds)

    client.join()


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed bagua launch")
    parser.add_argument(
        "--host_list",
        default=None,
        help="Target node list, baguarun will access each node via ssh."
        "`--host_list` has two forms. The first, comma-separated domain name "
        "or IP, such as `--host_list 192.168.1.1,192.168.1.2`. In this case. the ssh port is "
        "specified by the `--ssh_port` argument. The second, domain name or IP with ssh "
        "port specified inline, such as `--host_list 192.168.1.1:22,192.168.1.2:8022`."
        "\nIf `--host_list` is None, baguarun will use the "
        "environment variable BAGUA_NODE_DOMAIN_NAMES.",
    )
    parser.add_argument(
        "--ssh_port",
        type=int,
        default=None,
        help="ssh port to use. you need to set it only when the ssh "
        "port of all nodes are the same. When "
        "ssh_port is not set, baguaru will use the "
        "environment variable BAGUA_SSH_PORT.",
    )
    parser.add_argument("--master_port", type=int, default=None)
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument(
        "--no_python",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--bagua_service_port",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--enable_bagua_net",
        action="store_true",
        default=False,
        help="Enable Bagua-Net optimization for better "
        "communication performance. See https://github.com/BaguaSys/bagua/tree/master/rust/bagua-net"
        " for more details.",
    )
    parser.add_argument(
        "-x",
        type=str,
        action="append",
        help="Environment variables that need to be passed in pssh.",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.host_list is None:
        args.host_list = os.environ.get("BAGUA_NODE_DOMAIN_NAMES", "")
    if args.ssh_port is None and "BAGUA_SSH_PORT" in os.environ:
        args.ssh_port = int(os.environ["BAGUA_SSH_PORT"])

    assert args.host_list, "`--host_list` or $BAGUA_NODE_DOMAIN_NAMES must be set"
    host_list = []
    for host_str in args.host_list.split(","):
        host_or_pair = host_str.split(":")
        if len(host_or_pair) == 1:
            assert args.ssh_port, "`--host_list` or $BAGUA_SSH_PORT must be set"
            host_list.append((host_or_pair[0], args.ssh_port))
        elif len(host_or_pair) == 2:
            (host, port) = host_or_pair
            host_list.append((host, int(port)))
        else:
            raise Exception("Invalid host={}".format(host_str))

    args.host_list = host_list

    args.set_env = {}
    if args.x is not None:
        for x_word in args.x:
            if "=" in x_word:
                (k, v) = x_word.split("=", 1)
                args.set_env[k] = v
            else:
                k = x_word
                args.set_env[k] = os.environ.get(k, "")

    return args


def main():
    args = parse_args()
    pssh_bagua_launch(
        args,
        script_cmd=" ".join(
            [
                args.training_script,
            ]
            + args.training_script_args
        ),
        env=args.set_env,
    )


if __name__ == "__main__":
    main()
