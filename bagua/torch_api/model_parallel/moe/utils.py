import torch


def is_moe_param(param: torch.Tensor) -> bool:
    """The MOE parameter should be skip allreduce.

    Args:
        param (torch.Tensor): torch parameter

    Returns:
        bool: [description]
    """
    if hasattr(param, "expert") and param.expert:
        return True
    return False
