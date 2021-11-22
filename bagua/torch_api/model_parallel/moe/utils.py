import torch


def is_moe_param(param: torch.Tensor) -> bool:
    """The MOE parameter should be skip allreduce.

    Args:
        param (torch.Tensor): torch parameter

    Returns:
        bool: [description]
    """
    dp_comm = None
    if hasattr(param, "dp_comm"):
        dp_comm = param.dp_comm
    else:
        dp_comm = "dp"

    if dp_comm == "none":
        return True

    if hasattr(param, "expert") and param.expert:
        return True
    return False
