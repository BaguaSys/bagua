#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm
from bagua.torch_api import get_world_size
from typing import List


class ByteGradAlgorithm(Algorithm):
    def __init__(self, average: bool = True):
        """
        Create an instance of the
        `ByteGrad <https://bagua-tutorials.kwai-seattle.com/algorithms/bytegrad>`_
        algorithm.

        Args:
            average (bool): If True, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.average = average

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        """
        Given the bucketing suggestion from Bagua, return the actual Bagua buckets.
        The default implementation follows the suggestion to do the bucketing.

        Args:
            tensors: Bagua tensors grouped in different
                lists, representing Bagua's suggestion on how to bucketing the
                tensors.

        Returns:
            A list of Bagua buckets.
        """
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(
                bucket, flatten=True, name=str(idx), alignment=get_world_size()
            )
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        bucket.append_centralized_synchronous_op(
            hierarchical=True,
            average=self.average,
            scattergather=True,
            compression="MinMaxUInt8",
        )
