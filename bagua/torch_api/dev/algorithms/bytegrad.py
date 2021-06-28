#!/usr/bin/env python3

from bagua.torch_api.dev.bucket import BaguaBucket
from bagua.torch_api.dev.distributed_dev import BaguaModule
from bagua.torch_api.dev.algorithms import Algorithm

class ByteGradAlgorithm(Algorithm):
    def __init__(self, average: bool=True):
        """
        Create an instance of the
        `ByteGrad <https://baguasys.github.io/tutorials/algorithms/bytegrad.html>`_
        algorithm.

        Args:
            average (bool): If True, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.average = average

    def init_operations(
            self,
            bagua_module: BaguaModule,
            bucket: BaguaBucket,
    ):
        bucket.backend_bucket.clear_ops()
        bucket.backend_bucket.append_centralized_synchronous_op(
            bagua_module.bagua_inter_node_communicator,
            bagua_module.bagua_intra_node_communicator,
            hierarchical=True,
            average=self.average,
            scattergather=True,
            compression="MinMaxUInt8",
        )
