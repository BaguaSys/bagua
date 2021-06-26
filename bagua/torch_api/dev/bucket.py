#!/usr/bin/env python3

from bagua.torch_api.dev.tensor import BaguaTensor
from typing import List


class BaguaBucket:
    def __init__(self, tensors: List[BaguaTensor]) -> None:
        self.tensors = tensors

    def register(self, bagua_backend):
        assert self.bagua_backend is None
        self.bagua_backend = bagua_backend
        ...

    def is_registered(self) -> bool:
        return not (self.bagua_backend is None)

    def flatten_(self):
        ...

    def is_flatten(self) -> bool:
        ...
