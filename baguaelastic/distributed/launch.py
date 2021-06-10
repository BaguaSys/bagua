#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021, Kuaishou AI Platform & DS3 Lab.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
This is a script forked from torchelastic, usage reference https://github.com/pytorch/elastic/blob/master/torchelastic/distributed/launch.py
"""
import logging
import os
import sys
import uuid
from argparse import REMAINDER, ArgumentParser

import bagua
import bagua.distributed.elastic.rendezvous.registry as rdzv_registry
from bagua.distributed.elastic import events
from bagua.distributed.elastic.multiprocessing import Std
from bagua.distributed.elastic.multiprocessing.errors import ChildFailedError, record
from bagua.distributed.elastic.rendezvous import RendezvousParameters
from bagua.distributed.elastic.rendezvous.etcd_server import EtcdServer
from bagua.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from bagua.distributed.elastic.utils.logging import get_logger
from torchelastic import metrics
from torchelastic.agent.server.api import WorkerSpec, WorkerState
from torchelastic.agent.server.local_elastic_agent import LocalElasticAgent
from torchelastic.distributed.argparse_util import check_env, env
import torch


log = get_logger()


def parse_args(args):
    """
    Helper function parsing the command line options.
    """

    parser = ArgumentParser(description="baguaelastic elastic training launcher")

    # Arguments for the launch helper
    # worker/node size related arguments
    parser.add_argument(
        "--nnodes",
        action=env,
        type=str,
        default="1:1",
        help="number of nodes or MIN_NODES:MAX_NODES",
    )
    parser.add_argument(
        "--nproc_per_node",
        action=env,
        type=str,
        default="auto",
        help="number of workers per node, supported values: [auto, cpu, gpu, int]",
    )

    # rendezvous related arguments
    parser.add_argument(
        "--rdzv_backend",
        action=env,
        type=str,
        default="etcd",
        help="rendezvous backend",
    )
    parser.add_argument(
        "--rdzv_endpoint",
        action=env,
        type=str,
        default="",
        help="rendezvous backend server host:port",
    )
    parser.add_argument("--rdzv_id", action=env, type=str, help="user defined group id")
    parser.add_argument(
        "--rdzv_conf",
        action=env,
        type=str,
        default="",
        help="additional rdzv configuration (conf1=v1,conf2=v2,...)",
    )

    # sidecar embed rdzv backend that defaults to etcd
    parser.add_argument(
        "--standalone",
        action=check_env,
        help="starts a local, standalone rdzv backend that is represented by"
        " etcd server on a random free port"
        "using the etcd binary specified in TORCHELASTIC_ETCD_BINARY_PATH"
        " env var or the one found in PATH."
        " Useful when launching single-node, multi-worker job."
        " If specified --rdzv_backend, --rdzv_endpoint, --rdzv_id"
        " are autoassigned, any explicitly set values are ignored",
    )

    # user-code launch related arguments
    parser.add_argument(
        "--max_restarts",
        action=env,
        type=int,
        default=3,
        help="max number of worker group restarts before failing",
    )
    parser.add_argument(
        "--monitor_interval",
        action=env,
        type=float,
        default=5,
        help="interval (in seconds) to monitor the state of workers",
    )
    parser.add_argument(
        "--start_method",
        action=env,
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="multiprocessing start_method to use when creating workers",
    )
    parser.add_argument(
        "--role",
        action=env,
        type=str,
        default="default",
        help="user-defined role for the workers",
    )
    parser.add_argument(
        "-m",
        "--module",
        action=check_env,
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        "'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        action=check_env,
        help='Do not prepend the training script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )

    parser.add_argument(
        "--log_dir",
        action=env,
        type=str,
        default=None,
        help="base dir to use for log files (e.g. /var/log/baguaelastic)"
        " can reuse the same dir for multiple runs "
        "(a unique job-level subdir is created with rdzv_id as the prefix)",
    )

    parser.add_argument(
        "-r",
        "--redirects",
        action=env,
        type=str,
        default="0",
        help="std streams to redirect into a log file in the log_dir"
        " (e.g. [-r 3] redirects both stdout+stderr for all workers,"
        " [-r 0:1,1:2] redirects stdout for local rank 0 and stderr for local rank 1)",
    )

    parser.add_argument(
        "-t",
        "--tee",
        action=env,
        type=str,
        default="0",
        help="tee std streams into a log file and also to console (see --redirects for format)",
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
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args(args)


def parse_min_max_nnodes(nnodes: str):
    arr = nnodes.split(":")

    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[1])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')

    return min_nodes, max_nodes


def determine_local_world_size(nproc_per_node: str):
    try:
        logging.info(f"Using nproc_per_node={nproc_per_node}.")
        return int(nproc_per_node)
    except ValueError:
        if nproc_per_node == "cpu":
            num_proc = os.cpu_count()
            device_type = "cpu"
        elif nproc_per_node == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available.")
            device_type = "gpu"
            num_proc = torch.cuda.device_count()
        elif nproc_per_node == "auto":
            if torch.cuda.is_available():
                num_proc = torch.cuda.device_count()
                device_type = "gpu"
            else:
                num_proc = os.cpu_count()
                device_type = "cpu"
        else:
            raise ValueError(f"Unsupported nproc_per_node value: {nproc_per_node}")

        log.info(
            f"Using nproc_per_node={nproc_per_node},"
            f" seting to {num_proc} since the instance "
            f"has {os.cpu_count()} {device_type}"
        )
        return num_proc


def _construct_event(args) -> events.Event:
    metadata = {
        "rdzv_backend": args.rdzv_backend,
        "run_id": args.run_id,
        "role": args.role,
    }
    return events.Event(
        name="baguaelastic.main", source=events.EventSource.AGENT, metadata=metadata
    )


@record
def main(args=None):
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    args = parse_args(args)
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    assert args.max_restarts >= 0

    elastic_agent = None

    if args.standalone:
        etcd_server = EtcdServer()
        etcd_server.start()
        args.rdzv_backend = "etcd"
        args.rdzv_endpoint = etcd_server.get_endpoint()
        args.rdzv_id = str(uuid.uuid4())
        log.info(
            f"\n**************************************\n"
            f"Rendezvous info:\n"
            f"--rdzv_backend={args.rdzv_backend} "
            f"--rdzv_endpoint={args.rdzv_endpoint} "
            f"--rdzv_id={args.rdzv_id}\n"
            f"**************************************\n"
        )

    nproc_per_node = determine_local_world_size(args.nproc_per_node)
    if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
        omp_num_threads = 1
        print(
            f"*****************************************\n"
            f"Setting OMP_NUM_THREADS environment variable for each process to be "
            f"{omp_num_threads} in default, to avoid your system being overloaded, "
            f"please further tune the variable for optimal performance in "
            f"your application as needed. \n"
            f"*****************************************"
        )
        # This env variable will be passed down to the subprocesses
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    with_python = not args.no_python
    cmd = []
    if with_python:
        cmd = [sys.executable, "-u"]
        if args.module:
            cmd.append("-m")
    else:
        if args.module:
            raise ValueError(
                "Don't use both the '--no_python' flag"
                " and the '--module' flag at the same time."
            )

    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    rdzv_parameters = RendezvousParameters(
        backend=args.rdzv_backend,
        endpoint=args.rdzv_endpoint,
        run_id=args.rdzv_id,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        **_parse_rendezvous_config(args.rdzv_conf),
    )

    rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_parameters)
    try:
        spec = WorkerSpec(
            role=args.role,
            local_world_size=nproc_per_node,
            entrypoint=cmd[0],
            args=(*cmd[1:],),
            rdzv_handler=rdzv_handler,
            max_restarts=args.max_restarts,
            monitor_interval=args.monitor_interval,
            redirects=Std.from_str(args.redirects),
            tee=Std.from_str(args.tee),
        )
        metrics.initialize_metrics()
        elastic_agent = LocalElasticAgent(
            spec=spec, start_method=args.start_method, log_dir=args.log_dir
        )
        run_result = elastic_agent.run(spec.role)
        events.record(elastic_agent.get_agent_status_event(WorkerState.SUCCEEDED))
        if run_result.is_failed():
            # ChildFailedError is treated specially by @record
            # if the error files for the failed children exist
            # @record will copy the first error (root cause)
            # to the error file of the launcher process
            raise ChildFailedError(
                name=args.training_script,
                failures=run_result.failures,
            )
    except ChildFailedError:
        raise
    except Exception:
        if elastic_agent:
            events.record(elastic_agent.get_agent_status_event(WorkerState.FAILED))
        else:
            events.record(_construct_event(args))
        raise
    finally:
        rdzv_handler.shutdown()
        if args.standalone:
            etcd_server.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
    )
    log.info(f"Running baguaelastic.distributed.launch with args: {sys.argv}")
    main()
