#!/usr/bin/env python3

from bagua.torch_api.dev.algorithms import Algorithm

class GradientAllReduceAlgorithm(Algorithm):
    def __init__(self, hierarchical_reduce: bool=False, average: bool = True):
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
            bagua_module,
            bucket,
    ):
        bucket.backend_bucket.append_centralized_synchronous_op(
            bagua_module.bagua_inter_node_communicator,
            bagua_module.bagua_intra_node_communicator,
            hierarchical=self.hierarchical_reduce,
            average=self.average,
        )
