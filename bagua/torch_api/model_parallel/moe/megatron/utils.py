import os
import torch
from collections import OrderedDict
from typing import Dict, Optional

from megatron import mpu


def _add_moe_args(parser):
    group = parser.add_argument_group(title="moe")

    group.add_argument(
        "--num-local-experts", type=int, default=0, help="num of local experts"
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="default=1, top-k gating value, only supports k=1 or k=2.",
    )

    return parser


def get_moe_checkpoint_name(
    checkpoints_path: str,
    iteration: int,
    release: Optional[bool] = False,
    data_parallel_rank: Optional[int] = -1,
) -> str:

    if data_parallel_rank == -1:
        data_parallel_rank = mpu.get_data_parallel_rank()
    if data_parallel_rank == 0:
        from megatron.checkpointing import get_checkpoint_name

        return get_checkpoint_name(checkpoints_path, iteration, release)

    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)
    # Use both the tensor and pipeline MP rank.
    if mpu.get_pipeline_model_parallel_world_size() == 1:
        return os.path.join(
            checkpoints_path,
            directory,
            "mp_rank_{:02d}_dp_rank_{:04d}".format(
                mpu.get_tensor_model_parallel_rank(), data_parallel_rank
            ),
            "model_optim_rng.pt",
        )
    return os.path.join(
        checkpoints_path,
        directory,
        "mp_rank_{:02d}_{:03d}_dp_rank_{:04d}".format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            data_parallel_rank,
        ),
        "model_optim_rng.pt",
    )


def merge_state_dict(
    state_dict_rank0: Dict[str, torch.Tensor],
    state_dict_local: Dict[str, torch.Tensor],
    fp16: bool,
) -> Dict[str, torch.Tensor]:
    """merge two state dicts, one from data parallel rank 0,
    another only contains expert states"""

    def merge_model(state_dict_rank0, state_dict_local):
        for k, v in state_dict_local.items():
            # megatron uses both dict and OrderedDict in its state_dict
            if isinstance(v, (OrderedDict, dict)):
                merge_model(state_dict_rank0[k], v)
            else:
                state_dict_rank0[k] = v

    merge_model(state_dict_rank0["model"], state_dict_local["model"])

    optimizer_rank0 = (
        state_dict_rank0["optimizer"]["optimizer"]
        if fp16
        else state_dict_rank0["optimizer"]
    )
    optimizer_local = (
        state_dict_local["optimizer"]["optimizer"]
        if fp16
        else state_dict_local["optimizer"]
    )

    for k, v in optimizer_local["state"].items():
        optimizer_rank0["state"][k] = v

    if fp16:
        for group_idx, param_group in enumerate(
            state_dict_local["optimizer"]["fp32_from_fp16_params"]
        ):
            for param_in_group_idx, param in enumerate(param_group):
                if param is not None:
                    state_dict_rank0["optimizer"]["fp32_from_fp16_params"][group_idx][
                        param_in_group_idx
                    ] = param

    return state_dict_rank0
