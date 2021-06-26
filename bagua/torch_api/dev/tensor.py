#!/usr/bin/env python3

import torch

class BaguaTensor(torch.Tensor):
    def __init__(self, original_tensor: torch.Tensor) -> None:
        super().__init__()
        self.inner = original_tensor
        self.bagua_backend = None

    def register(self, bagua_backend):
        assert self.bagua_backend is None
        self.bagua_backend = bagua_backend
        ...

    def is_registered(self) -> bool:
        return not (self.bagua_backend is None)
