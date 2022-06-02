# Copyright (c) 2021 Kuaishou AI Platform & DS3 Lab
#
# All rights reserved.
#
# The file has been adapted from Megatron
#  https://github.com/NVIDIA/Megatron-LM/blob/v2.6/megatron/checkpointing.py
# Git commit hash: 3860e995269df61d234ed910d4756e104e1ab844
# We retain the following license from the original files:

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#

"""Input/output checkpointing."""

import os
import random
import sys
import torch
import bagua.torch_api as bagua
import numpy as np
from collections import OrderedDict
from typing import Optional, Union
from torch.nn.parallel import DistributedDataParallel as torchDDP
from bagua.torch_api.model_parallel.moe.megatron.utils import get_moe_checkpoint_name
from bagua.torch_api.model_parallel.moe.megatron.utils import merge_state_dict


from megatron import get_args, mpu, print_rank_0, print_rank_last

from megatron.checkpointing import (
    check_checkpoint_args,
    ensure_directory_exists,
    get_checkpoint_version,
    get_checkpoint_tracker_filename,
    set_checkpoint_version,
    update_num_microbatches,
)


def save_checkpoint(
    iteration: int,
    model: Union[torchDDP],
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    version22: Optional[bool] = False,
):
    """Save a model checkpoint."""
    args = get_args()
    if not torch.distributed.is_initialized() or mpu.get_data_parallel_rank() == 0:
        from megatron.checkpointing import save_checkpoint as save_checkpoint_megatron

        save_checkpoint_megatron(iteration, model, optimizer, lr_scheduler)
        return

    if args.num_local_experts > 0:
        _save_checkpoint_moe(iteration, model, optimizer, lr_scheduler, version22)
        return

    # it's necessary to barrier 2 times
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.barrier()


def _save_checkpoint_moe(
    iteration: int,
    model: Union[torchDDP],
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    version22: Optional[bool] = False,
):
    args = get_args()
    if version22:
        if hasattr(model, "module"):
            model = model.module
    else:
        from megatron import utils

        model = utils.unwrap_model(model)

    # Arguments, iteration, and model.
    state_dict = {}
    if version22:
        state_dict["model"] = model.state_dict_for_save_checkpoint(
            keep_vars=(mpu.get_data_parallel_rank() > 0)
        )
    else:
        if len(model) == 1:
            state_dict["model"] = model[0].state_dict_for_save_checkpoint(
                keep_vars=(mpu.get_data_parallel_rank() > 0)
            )
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict["model%d" % i] = model[i].state_dict_for_save_checkpoint()

    def extract_expert_param(state_dict):
        state_dict_new = state_dict.__class__()
        for k, v in state_dict.items():
            # megatron uses both dict and OrderedDict in its state_dict
            if isinstance(v, (OrderedDict, dict)):
                v_new = extract_expert_param(v)
                if len(v_new) > 0:
                    state_dict_new[k] = v_new
            elif bagua.moe.is_moe_param(v):
                state_dict_new[k] = v.detach()
        return state_dict_new

    state_dict["model"] = extract_expert_param(state_dict["model"])

    # Optimizer stuff.
    if not args.no_save_optim:
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
            param_global_idx = 0
            # TODO version26 state_dict["optimizer"]["optimizer"]["state"] is empty?
            for param_group in optimizer.optimizer.param_groups:
                for param in param_group["params"]:
                    if not bagua.moe.is_moe_param(param):
                        # this parameter is not an expert parameter
                        # thus there is no need to save its state in current rank
                        # since it has been saved by data parallel rank 0
                        if args.fp16:
                            # fp16 optimizer may have empty state due to overflow
                            state_dict["optimizer"]["optimizer"]["state"].pop(
                                param_global_idx, None
                            )
                        else:
                            state_dict["optimizer"]["state"].pop(param_global_idx)
                    param_global_idx += 1
            if args.fp16:
                state_dict["optimizer"]["optimizer"].pop("param_groups")
                # fp32_from_fp16_params in state_dict is not a copy
                # but a reference to optimizer.fp32_from_fp16_params,
                # changing it in state_dict will change
                # optimizer.fp32_from_fp16_params as well
                # thus we create an empty fp32_from_fp16_params in state_dict
                # and only insert expert parameters.
                fp32_from_fp16_params = state_dict["optimizer"]["fp32_from_fp16_params"]
                state_dict["optimizer"]["fp32_from_fp16_params"] = []
                for param_group in fp32_from_fp16_params:
                    param_group_copy = []
                    for param in param_group:
                        param_copy = param if bagua.moe.is_moe_param(param) else None
                        param_group_copy.append(param_copy)
                    state_dict["optimizer"]["fp32_from_fp16_params"].append(
                        param_group_copy
                    )
            else:
                state_dict["optimizer"].pop("param_groups")

    checkpoint_name = get_moe_checkpoint_name(args.save, iteration)
    ensure_directory_exists(checkpoint_name)
    torch.save(state_dict, checkpoint_name)

    # it's necessary to barrier 2 times
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.barrier()


def load_checkpoint(
    model: Union[torchDDP],
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    load_arg: Optional[str] = "load",
    strict: Optional[bool] = True,
    version22: Optional[bool] = False,
) -> int:
    args = get_args()
    if (
        not torch.distributed.is_initialized()
        or mpu.get_data_parallel_rank() == 0
        or args.num_local_experts == 0
    ):
        from megatron.checkpointing import load_checkpoint as load_checkpoint_megatron

        if version22:
            return load_checkpoint_megatron(model, optimizer, lr_scheduler, load_arg)
        else:
            return load_checkpoint_megatron(
                model, optimizer, lr_scheduler, load_arg, strict
            )

    return _load_checkpoint_moe(
        model, optimizer, lr_scheduler, load_arg, strict, version22
    )


def _load_checkpoint_moe(
    model: Union[torchDDP],
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    load_arg: Optional[str] = "load",
    strict: Optional[bool] = True,
    version22: Optional[bool] = False,
) -> int:
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    load_dir = getattr(args, load_arg)

    if version22:
        if hasattr(model, "module"):
            model = model.module
    else:
        from megatron import utils

        model = utils.unwrap_model(model)

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return iretation zero.
    if not os.path.isfile(tracker_filename):
        print_rank_0(
            "WARNING: could not find the metadata file {} ".format(tracker_filename)
        )
        print_rank_0("    will not load any checkpoints and will start from " "random")
        return 0

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    if version22:
        iteration = 0
        release = False
        with open(tracker_filename, "r") as f:
            metastring = f.read().strip()
            try:
                iteration = int(metastring)
            except ValueError:
                release = metastring == "release"
                if not release:
                    print_rank_last(
                        "ERROR: Invalid metadata file {}. Exiting".format(
                            tracker_filename
                        )
                    )
                    sys.exit()
        assert iteration > 0 or release, "error parsing metadata file {}".format(
            tracker_filename
        )
    else:
        from megatron.checkpointing import read_metadata

        iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    checkpoint_name_rank0 = get_moe_checkpoint_name(load_dir, iteration, release, 0)
    checkpoint_name_local = get_moe_checkpoint_name(
        load_dir, iteration, release, mpu.get_data_parallel_rank()
    )
    print_rank_last(
        " loading checkpoint at rank 0 from {} and rank {} from {} at iteration {}, will merge them later".format(
            checkpoint_name_rank0,
            mpu.get_data_parallel_rank(),
            checkpoint_name_local,
            iteration,
        )
    )

    # Load the checkpoint.
    def load_state_dict(checkpoint_name):
        try:
            state_dict = torch.load(checkpoint_name, map_location="cpu")
        except ModuleNotFoundError:
            # For backward compatibility.
            print_rank_last(" > deserializing using the old code structure ...")
            sys.modules["fp16.loss_scaler"] = sys.modules[
                "megatron.fp16_deprecated.loss_scaler"
            ]
            sys.modules["megatron.fp16.loss_scaler"] = sys.modules[
                "megatron.fp16_deprecated.loss_scaler"
            ]
            state_dict = torch.load(checkpoint_name, map_location="cpu")
            sys.modules.pop("fp16.loss_scaler", None)
            sys.modules.pop("megatron.fp16.loss_scaler", None)
        return state_dict

    state_dict_rank0 = load_state_dict(checkpoint_name_rank0)
    state_dict_local = load_state_dict(checkpoint_name_local)

    state_dict = merge_state_dict(state_dict_rank0, state_dict_local, args.fp16)

    # set checkpoint version
    set_checkpoint_version(state_dict.get("checkpoint_version", 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict["iteration"]
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict["total_iters"]
            except KeyError:
                print_rank_0(
                    "A metadata file exists but unable to load "
                    "iteration from checkpoint {}, exiting".format(
                        checkpoint_name_local
                    )
                )
                sys.exit()

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if "args" in state_dict:
        checkpoint_args = state_dict["args"]
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(
            checkpoint_args, "consumed_train_samples", 0
        )
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(
            checkpoint_args, "consumed_valid_samples", 0
        )
    else:
        print_rank_last("could not find arguments in the checkpoint ...")

    # Model.
    if version22:
        model.load_state_dict(state_dict["model"])
    else:
        if len(model) == 1:
            model[0].load_state_dict(state_dict["model"], strict=strict)
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                model[i].load_state_dict(state_dict["model%d" % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f" checkpoint version {checkpoint_version}")
    if not version22:
        from megatron.checkpointing import fix_query_key_value_ordering

        fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            if optimizer is not None:
                optimizer.load_state_dict(state_dict["optimizer"])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        except KeyError:
            print_rank_last(
                "Unable to load optimizer from checkpoint {}. "
                "Specify --no-load-optim or --finetune to prevent "
                "attempting to load the optimizer state, "
                "exiting ...".format(checkpoint_name_local)
            )
            sys.exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(state_dict["random_rng_state"])
            np.random.set_state(state_dict["np_rng_state"])
            torch.set_rng_state(state_dict["torch_rng_state"])
            torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            # Check for empty states array
            if not state_dict["rng_tracker_states"]:
                raise KeyError
            mpu.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])
        except KeyError:
            print_rank_last(
                "Unable to load rng state from checkpoint {}. "
                "Specify --no-load-rng or --finetune to prevent "
                "attempting to load the rng state, "
                "exiting ...".format(checkpoint_name_local)
            )
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_last(
        "  successfully loaded checkpoint (with expert parametes updated) from {} at iteration {}".format(
            args.load, iteration
        )
    )

    return iteration
