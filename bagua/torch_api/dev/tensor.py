#!/usr/bin/env python3

import torch

class BaguaTensor(torch.Tensor):
    def __init__(self, original_tensor: torch.Tensor) -> None:
        super().__init__()
        self.inner = original_tensor
