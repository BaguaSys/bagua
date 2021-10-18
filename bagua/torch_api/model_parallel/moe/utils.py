import torch


def is_moe_param(param: torch.Tensor) -> bool:
    if hasattr(param, "expert") and param.expert:
        return True
    return False
