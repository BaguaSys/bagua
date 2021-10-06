#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api import get_world_size
from typing import List
from bagua.torch_api.algorithms_implementation.bytegrad_implementation import (
    ByteGradAlgorithm_Implementation,
)


class ByteGradAlgorithm:
    def __init__(self, average: bool = True):
        """
        Create an instance of the
        `ByteGrad <https://bagua-tutorials.kwai-seattle.com/algorithms/bytegrad>`_
        algorithm.

        Args:
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.average = average

    def reify(self) -> ByteGradAlgorithm_Implementation:
        return ByteGradAlgorithm_Implementation(
            average=self.average,
        )
