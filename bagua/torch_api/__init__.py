#!/usr/bin/env python3
"""
The Bagua communication library PyTorch interface.
"""
import time
from enum import Enum
from .exceptions import UnsupportedAlgorithmException
from .utils import apply_flattened_call, flatten
from .distributed import OverlappingWrapper, BucketType, DistributedModule
from .distributed_define import ReduceOp
from .communication import (
    is_initialized,
    init_process_group,
    broadcast_coalesced,
    broadcast,
    allreduce_coalesced,
    allreduce,
    get_bagua_hyperparameters,
    get_hyperparameters_service_client,
)
from .env import (
    get_rank,
    get_world_size,
    get_local_rank,
    get_local_size,
    get_autotune_server_addr,
    is_report_metrics_switch_on,
    get_autotune_level,
    _horovod_0_21_1_compat_mode,
    _horovod_0_21_3_compat_mode,
)
from .algorithms.decentralize import DecentralizedReducer
from bagua.torch_api.algorithms.allreduce import Allreducer, ScatterGatherAllreducer
from .fuse_optimizer import FusedOptimizer
from .compression import Compressor
import torch
from . import tools
from bagua.bagua_define import DistributedAlgorithm, BaguaHyperparameter
from bagua.torch_api.utils import average_by_removing_extreme_values
import logging
from typing import List
import copy
from typing import Union


class ModelSwitchWrapper(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        delay_reduce: bool = False,
        hierarchical_reduce: Union[bool, None] = None,
        message_size: int = 10_000_000,
        intra_comm_root_rank: int = 0,
        **kwargs,
    ):
        super(ModelSwitchWrapper, self).__init__()

        self.module = module
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizers = [optimizer]
        else:
            optimizers = optimizer

        self.optimizers = optimizers
        self.delay_reduce = delay_reduce
        self.hierarchical_reduce = hierarchical_reduce
        self.message_size = message_size
        self.intra_comm_root_rank = intra_comm_root_rank
        self.kwargs: dict = kwargs

        self.bagua_module: torch.nn.Module = torch.nn.Module()
        self.stage = 0
        self.autotune_client = get_hyperparameters_service_client()
        self.step_counter = 0

        self.last_hit_tp = time.time()
        self.score_record_list: List[float] = []
        self.autotune_flag = get_autotune_level() >= 1

    def switch_to(
        self,
        distributed_algorithm: Union[DistributedAlgorithm, str],
    ):
        # Reset Autotune Server
        self.autotune_client.reset()
        self.bagua_module = self.module
        if isinstance(distributed_algorithm, str):
            distributed_algorithm = DistributedAlgorithm.from_str(distributed_algorithm)

        # sync params at the start of each training stage
        if self.stage == 0:
            if _horovod_0_21_1_compat_mode():
                from .horovod_pack import initialize_optimizer_state_0_21_1

                initialize_optimizer_state_0_21_1(self.optimizers[0])
            elif _horovod_0_21_3_compat_mode():
                from .horovod_pack import initialize_optimizer_state_0_21_3

                initialize_optimizer_state_0_21_3(self.optimizers[0])

            broadcast_parameters(self.bagua_module)
        else:
            allreduce_parameters(self.bagua_module)

        self.stage += 1
        # Initialize distributed module reducer
        if distributed_algorithm == DistributedAlgorithm.GradientAllReduce:
            self.bagua_module = Allreducer(self.bagua_module, **self.kwargs)
            self.bagua_module = OverlappingWrapper(
                self.bagua_module,
                self.optimizers,
                delay_reduce=self.delay_reduce,
                hierarchical_reduce=(
                    False
                    if self.hierarchical_reduce is None
                    else self.hierarchical_reduce
                ),
                message_size=self.message_size,
                chunking=False,
                **self.kwargs,
            )
        elif distributed_algorithm == DistributedAlgorithm.ScatterGatherAllReduce:
            self.bagua_module = ScatterGatherAllreducer(
                self.bagua_module, **self.kwargs
            )
            self.bagua_module = OverlappingWrapper(
                self.bagua_module,
                self.optimizers,
                delay_reduce=self.delay_reduce,
                hierarchical_reduce=(
                    True
                    if self.hierarchical_reduce is None
                    else self.hierarchical_reduce
                ),
                message_size=self.message_size,
                chunking=True,
                **self.kwargs,
            )
        elif distributed_algorithm == DistributedAlgorithm.Decentralize:
            self.bagua_module = DecentralizedReducer(self.bagua_module, **self.kwargs)
            self.bagua_module = OverlappingWrapper(
                self.bagua_module,
                self.optimizers,
                delay_reduce=self.delay_reduce,
                bucket_type=BucketType.Weight,
                message_size=self.message_size,
                hierarchical_reduce=(
                    True
                    if self.hierarchical_reduce is None
                    else self.hierarchical_reduce
                ),
                decentralize_reduce=True,
                chunking=False,
                **self.kwargs,
            )
        elif distributed_algorithm == DistributedAlgorithm.QuantizeAllReduce:
            self.bagua_module = ScatterGatherAllreducer(
                self.bagua_module, compressor=Compressor.Uint8Compressor, **self.kwargs
            )
            self.bagua_module = OverlappingWrapper(
                self.bagua_module,
                self.optimizers,
                delay_reduce=self.delay_reduce,
                hierarchical_reduce=(
                    True
                    if self.hierarchical_reduce is None
                    else self.hierarchical_reduce
                ),
                message_size=self.message_size,
                chunking=True,
                **self.kwargs,
            )
        else:
            raise UnsupportedAlgorithmException(distributed_algorithm)

        get_bagua_hyperparameters().update(
            {
                "distributed_algorithm": distributed_algorithm.value,  # type: ignore
                "is_hierarchical_reduce": bool(self.hierarchical_reduce),
            }
        )
        # TODO: Configure the default hyperparameters for different algorithms

        return self

    def state_dict(self, **kwargs):
        return self.module.state_dict(**kwargs)

    def report_metrics(self, score_record_list):
        if len(score_record_list) == 0:
            iter_per_seconds = 0.0
        else:
            iter_per_seconds = sum(score_record_list) / len(score_record_list)
        logging.info("score_record_list={}".format(score_record_list))
        denoised_iter_per_seconds, std, _ = average_by_removing_extreme_values(
            score_record_list
        )
        logging.info(
            "iter_per_seconds={}, denoised_iter_per_seconds={}, std={}".format(
                iter_per_seconds,
                denoised_iter_per_seconds,
                std,
            )
        )

        self.autotune_client.report_metrics(
            rank=get_rank(),
            unix_timestamp=time.time(),
            train_iter=self.step_counter,
            iter_per_seconds=iter_per_seconds,
            denoised_iter_per_seconds=denoised_iter_per_seconds,
            hyperparameters=get_bagua_hyperparameters().dict(),
        )

    def ask_and_update_hyperparameters(self) -> bool:
        rsp = self.autotune_client.ask_hyperparameters(
            rank=get_rank(), train_iter=self.step_counter
        )
        recommended_hyperparameters = copy.deepcopy(get_bagua_hyperparameters()).update(
            rsp.json()["recommended_hyperparameters"]
        )
        self.autotune_flag = rsp.json()["is_autotune_processing"]

        def update_hyperparameters(rec_hp):
            my_hp = get_bagua_hyperparameters()
            logging.info(
                "rec_hp={}, my_hp={}, rec_hp.buckets={}".format(
                    rec_hp.dict(), my_hp.dict(), rec_hp.buckets
                )
            )
            if rec_hp.distributed_algorithm != my_hp.distributed_algorithm:
                # The proposal to replace the communication algorithm is currently not supported
                return False

            if (
                DistributedAlgorithm(rec_hp.distributed_algorithm)
                is DistributedAlgorithm.GradientAllReduce
            ):
                rec_kwargs = {}
                if rec_hp.buckets and rec_hp.buckets != my_hp.buckets:
                    rec_kwargs["buckets"] = rec_hp.buckets
                if rec_hp.is_hierarchical_reduce != my_hp.is_hierarchical_reduce:
                    rec_kwargs["hierarchical_reduce"] = rec_hp.is_hierarchical_reduce

                if rec_kwargs:
                    logging.info("update hyperparameters to {}".format(rec_kwargs))
                    self.bagua_module.reset_reducer(**rec_kwargs)
                    get_bagua_hyperparameters().update(rec_hp.dict())
            elif (
                DistributedAlgorithm(rec_hp.distributed_algorithm)
                is DistributedAlgorithm.Decentralize
            ):
                # TODO: support decentralize autotune
                return False

            return True

        return update_hyperparameters(recommended_hyperparameters)

    def forward(self, *inputs, **kwargs):
        assert self.bagua_module is not None
        result = self.bagua_module(*inputs, **kwargs)

        if self.module.training:
            cycle_step = 100
            if self.step_counter != 0 and self.step_counter % cycle_step == 0:
                st = time.time()

                if get_autotune_level() >= 1 and self.autotune_flag:
                    self.score_record_list.append(
                        cycle_step / float(time.time() - self.last_hit_tp)
                    )

                    self.report_metrics(self.score_record_list)
                    whether_the_parameter_updated = (
                        self.ask_and_update_hyperparameters()
                    )
                    if whether_the_parameter_updated:
                        self.score_record_list.clear()

                    self.last_hit_tp = time.time()

                logging.info("autotune overhead={}".format(time.time() - st))

            self.step_counter += 1

        return result


def _get_module_params_and_buffers(module):
    if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
        parameters_to_ignore = module._ddp_params_and_buffers_to_ignore
    else:
        parameters_to_ignore = []

    module_states = []
    for name, param in module.state_dict().items():
        if name not in parameters_to_ignore:
            module_states.append(param)

    return module_states


def broadcast_parameters(module):
    from .communication import _get_global_state

    module_states = _get_module_params_and_buffers(module)

    authoritative_rank = 0
    for state in module_states:
        broadcast(state, root=authoritative_rank)


def allreduce_parameters(module):
    from .communication import _get_global_state

    module_states = _get_module_params_and_buffers(module)

    for state in module_states:
        allreduce(state, average=True)


def bagua_init(
    module: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    distributed_algorithm: Union[
        DistributedAlgorithm, str
    ] = DistributedAlgorithm.GradientAllReduce,
    delay_reduce: bool = False,
    hierarchical_reduce: Union[bool, None] = None,
    message_size: int = 10_000_000,
    **kwargs,
):
    """
    `bagua_init` is a module wrapper that enables easy multiprocess distributed data parallel
    training using different distributed algorithms.

    Parameters are broadcast across participating processes on initialization, and gradients or
    weights are allreduced and averaged over processes during `backward()`.

    Arguments:
        * `module`(_torch.nn.Module_) - Network definition to be run in multi-gpu/distributed mode.
        * `distributed_algorithm`(_DistributedAlgorithm_) - Distributed algorithm used to average
           gradients or weights across all workers. Default: `DistributedAlgorithm.GradientAllReduce`.
        * `delay_reduce`(_bool_) - Overlap communication with computation. Default: `True`.
        * `delay_reduce`(_bool_): Delay all communication to the end of the backward pass. This disables
           overlapping communication with computation. Default value is `False`.
        * `hierarchical_reduce`(_bool_): Enable hierarchical reduce. For `GradientAllReduce` algorithm, default
           value is `False`, otherwise, default value is `True`.
        * `message_size`(_int_) - Minimum bytes in a communication bucket. Default: `10_000_000`.

    Yields:
        Distributed module.

    Examples:

        ```python
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model, optimizer = bagua_init(model, optimizer)
        ```
    """

    assert is_initialized(), "Must call bagua.init_process_group() first!"

    # Set DistributedAlgorithm Wrapper
    module = ModelSwitchWrapper(
        module=module,
        optimizer=optimizer,
        delay_reduce=delay_reduce,
        hierarchical_reduce=hierarchical_reduce,
        message_size=message_size,
        **kwargs,
    ).switch_to(distributed_algorithm)

    return module, optimizer
