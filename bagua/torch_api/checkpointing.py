# Copyright (c) 2021 Kuaishou AI Platform & DS3 Lab
#
# All rights reserved.

import logging
import os
import re
import sys
import torch
import torch.distributed as dist
from collections import defaultdict


def _ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def _get_optimizer_ckpt_name(
    checkpoints_path, iteration, expp_rank, mp_rank=0, release=False
):
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)
    ckpt_name = os.path.join(
        checkpoints_path,
        directory,
        f"expp_rank_{expp_rank}_mp_rank_{mp_rank:02d}_optim_states.pt",
    )
    return ckpt_name


def _get_expert_ckpt_name(
    checkpoints_path, expert_id, iteration, mp_rank=0, release=False
):
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)
    ckpt_name = os.path.join(
        checkpoints_path,
        directory,
        f"expert_{expert_id}_mp_rank_{mp_rank:02d}_model_states.pt",
    )
    return ckpt_name


def _get_model_ckpt_name(checkpoints_path, iteration, mp_rank=0, release=False):
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)
    return os.path.join(
        checkpoints_path, directory, f"mp_rank_{mp_rank:02d}_model_states.pt"
    )


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, "latest_checkpointed_iteration.txt")


def read_metadata(tracker_filename):
    iteration = 0
    release = False
    with open(tracker_filename, "r") as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == "release"
            if not release:
                logging.error(
                    "Invalid metadata file {}. Exiting".format(tracker_filename)
                )
                sys.exit()
    assert iteration >= 0 or release, "error parsing metadata file {}".format(
        tracker_filename
    )

    return iteration, release


def save_checkpoint(
    iteration, checkpoints_path, model, optimizer=None, lr_scheduler=None
):
    logging.info(
        "saving checkpoint at iterration {:7d} to {}".format(
            iteration, checkpoints_path
        )
    )

    if model.has_moe_layers:
        _save_moe_checkpoint(
            iteration, checkpoints_path, model, optimizer, lr_scheduler
        )
    else:
        _save_checkpoint(iteration, checkpoints_path, model, optimizer, lr_scheduler)

    logging.info(
        "successfully saved checkpoint at iterration {:7d} to {}".format(
            iteration, checkpoints_path
        )
    )

    # update the latest iteration
    if not dist.is_initialized() or dist.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(checkpoints_path)
        with open(tracker_filename, "w") as f:
            f.write(str(iteration))


def _save_checkpoint(
    iteration, checkpoints_path, model, optimizer=None, lr_scheduler=None
):
    if not dist.is_initialized() or dist.get_rank() == 0:
        state_dict = {}
        state_dict["iteration"] = iteration
        state_dict["model"] = model.state_dict()
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            state_dict["lr_scheduler"] = lr_scheduler.state_dict()

        checkpoint_name = _get_model_ckpt_name(checkpoints_path, iteration)
        _ensure_directory_exists(checkpoint_name)
        torch.save(state_dict, checkpoint_name)

    if dist.is_initialized():
        dist.barrier()


def _save_moe_checkpoint(
    iteration, checkpoints_path, model, optimizer=None, lr_scheduler=None
):
    world_size = 1 if not dist.is_initialized() else dist.get_world_size()
    expp_rank = 1 if not dist.is_initialized() else dist.get_rank()
    num_local_experts = model.num_experts // world_size
    experts_state_dict, model_state_dict = _get_moe_state_dict(
        model.state_dict(), num_local_experts, expp_rank
    )

    #  Each rank saves its local experts
    for global_expert_id, expert_state_dict in experts_state_dict.items():
        expert_save_dir = _get_expert_ckpt_name(
            checkpoints_path, global_expert_id, iteration
        )
        logging.info(
            f"Saving model expert {global_expert_id} checkpoint: {expert_save_dir}"
        )
        _ensure_directory_exists(expert_save_dir)
        torch.save(expert_state_dict, expert_save_dir)

    # Save optimizer states. They are different across each exp parallel rank.
    optimizer_state = {"optimizer": optimizer.state_dict() if optimizer else None}
    torch.save(
        optimizer_state,
        _get_optimizer_ckpt_name(checkpoints_path, iteration, expp_rank),
    )

    if expp_rank == 0:
        state_dict = {}
        state_dict["iteration"] = iteration
        state_dict["model"] = model_state_dict
        if lr_scheduler is not None:
            state_dict["lr_scheduler"] = lr_scheduler.state_dict()

        # Save.
        checkpoint_name = _get_model_ckpt_name(checkpoints_path, iteration)
        _ensure_directory_exists(checkpoint_name)
        torch.save(state_dict, checkpoint_name)


def _get_moe_state_dict(full_state_dict, num_local_experts, expp_rank):
    experts_state_dict, moe_state_dict = defaultdict(dict), {}
    for key in list(full_state_dict.keys()):
        if "expert" in key and "moe.gate.wg.weight" not in key:
            moe_state_dict[key] = full_state_dict.pop(key)
    non_moe_state_dict = full_state_dict

    moe_str_prefix = ".bagua_moe.experts.bagua_experts."
    for key in list(moe_state_dict.keys()):
        m = re.match(f".*{moe_str_prefix}([0-9]+).*", key)
        local_expert_id = None
        if not m:
            logging.warning(f"No expert found in key {key}.")
        else:
            local_expert_id = m.group(1)

        global_expert_id = expp_rank * num_local_experts + int(local_expert_id)
        expert_key = key.replace(
            f"{moe_str_prefix}{local_expert_id}", f"{moe_str_prefix}{global_expert_id}"
        )
        experts_state_dict[str(global_expert_id)][expert_key] = moe_state_dict.pop(key)

    return experts_state_dict, non_moe_state_dict


def load_checkpoint(
    checkpoints_path, model, optimizer=None, lr_scheduler=None, strict=True
):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """

    tracker_filename = get_checkpoint_tracker_filename(checkpoints_path)

    # If no tracker file, return iretation zero.
    if not os.path.isfile(tracker_filename):
        logging.warning(f"could not find checkpoint metadata file {tracker_filename}")
        return 0

    iteration, release = read_metadata(tracker_filename)
    logging.info(f"loading checkpoint from {checkpoints_path} at iteration {iteration}")

    _load_checkpoint(iteration, checkpoints_path, model, optimizer, lr_scheduler)

    if dist.is_initialized():
        dist.barrier()

    logging.info(
        f"successfully loaded checkpoint from {checkpoints_path} at {iteration}"
    )
    return iteration


def _load_checkpoint(
    iteration, checkpoints_path, model, optimizer=None, lr_scheduler=None, strict=True
):
    expp_rank = 1 if not dist.is_initialized() else dist.get_rank()
    checkpoint_name = _get_model_ckpt_name(checkpoints_path, iteration)

    model_checkpoint = torch.load(checkpoint_name, map_location="cpu")
    if model.has_moe_layers:
        num_local_experts = model.num_experts // dist.get_world_size()
        _load_moe_state_dict(
            checkpoints_path,
            iteration,
            num_local_experts,
            expp_rank,
            state_dict=model_checkpoint["model"],
        )

    if model.has_moe_layers and optimizer is not None:
        optim_load_path = _get_optimizer_ckpt_name(
            checkpoints_path, iteration, expp_rank
        )
        optim_checkpoint = torch.load(optim_load_path, map_location=torch.device("cpu"))
    else:
        optim_checkpoint = model_checkpoint

    model.load_state_dict(model_checkpoint["model"], strict=strict)

    # Optimizer.
    if optimizer is not None:
        optimizer.load_state_dict(optim_checkpoint["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(model_checkpoint["lr_scheduler"])


def _load_moe_state_dict(
    checkpoint_path, iteration, num_local_experts, expp_rank, state_dict
):
    for local_expert_id in range(num_local_experts):
        global_expert_id = expp_rank * num_local_experts + local_expert_id
        expert_state_dict = torch.load(
            _get_expert_ckpt_name(checkpoint_path, global_expert_id, iteration),
            map_location=torch.device("cpu"),
        )

        # Updating global -> local expert ids
        moe_str_prefix = ".bagua_moe.experts.bagua_experts."
        for key in list(expert_state_dict.keys()):
            local_key = key.replace(
                f"{moe_str_prefix}{global_expert_id}",
                f"{moe_str_prefix}{local_expert_id}",
            )
            expert_state_dict[local_key] = expert_state_dict.pop(key)
        state_dict.update(expert_state_dict)
