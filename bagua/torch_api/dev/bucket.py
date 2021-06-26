#!/usr/bin/env python3

from bagua.torch_api.dev.tensor import BaguaTensor
from typing import List
import torch


class BaguaBucket:
    def __init__(self, tensors: List[BaguaTensor]) -> None:
        self.tensors = tensors
        self.backend_tensor = None


    def flatten_(self):
        # TODO: finish me @shaoduo
        """
        flatten inner tensors in place
        """

    def is_flatten(self) -> bool:
        # TODO: finish me @shaoduo
        ...
