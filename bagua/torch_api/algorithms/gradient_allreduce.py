#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms.base import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup


class GradientAllReduceAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = False,
        average: bool = True,
    ):
        """
        Implementation of the
        `GradientAllReduce <https://tutorials.baguasys.com/algorithms/gradient-allreduce>`_
        algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        super(GradientAllReduceAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.average = average

    def init_operations(
        self,
        _: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        bucket.append_centralized_synchronous_op(
            hierarchical=self.hierarchical,
            average=self.average,
            group=self.process_group,
        )


class GradientAllReduceAlgorithm(Algorithm):
    def __init__(self, hierarchical: bool = False, average: bool = True):
        """
        Create an instance of the
        `GradientAllReduce <https://tutorials.baguasys.com/algorithms/gradient-allreduce>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            average (bool): If ``True``, the gradients on each worker are averaged.
                Otherwise, they are summed.
        """
        self.hierarchical = hierarchical
        self.average = average

    def reify(self, process_group: BaguaProcessGroup) -> GradientAllReduceAlgorithmImpl:
        return GradientAllReduceAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            average=self.average,
        )
