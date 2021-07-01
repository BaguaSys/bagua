#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm


class GradientAllReduceAlgorithm(Algorithm):
    def __init__(self, hierarchical_reduce: bool = False, average: bool = True):
        """
        Create an instance of the
        `GradientAllReduce <https://baguasys.github.io/tutorials/algorithms/gradient-allreduce.html>`_
        algorithm.

        Args:
            hierarchical_reduce (bool): Enable hierarchical communication.
            average (bool): If True, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.hierarchical_reduce = hierarchical_reduce
        self.average = average

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):
        bucket.backend_bucket.clear_ops()
        if self.hierarchical_reduce:
            bucket.backend_bucket.append_centralized_synchronous_op(
                bagua_module.bagua_inter_node_communicator,
                bagua_module.bagua_intra_node_communicator,
                hierarchical=self.hierarchical_reduce,
                average=self.average,
            )
        else:
            bucket.backend_bucket.append_centralized_synchronous_op(
                bagua_module.bagua_global_communicator,
                None,
                hierarchical=self.hierarchical_reduce,
                average=self.average,
            )
