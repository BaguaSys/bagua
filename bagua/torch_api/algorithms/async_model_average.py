#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm
from typing import List
from bagua.torch_api.tensor import BaguaTensor

class AsyncModelAverageAlgorithm(Algorithm):
    def __init__(self, sync_interval_ms: int):
        """
        Create an instance of the
        `AsyncModelAverage <https://baguasys.github.io/tutorials/algorithms/async-model-average.html>`_
        algorithm.

        Args:
            sync_interval_ms (int): How many milliseconds between two model synchronizations.
        """
        self.sync_interval_ms = sync_interval_ms

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        parameters = bagua_module.bagua_build_params()
        tensors = [
           param.ensure_bagua_tensor(name)
            for name, param in parameters.__reversed__()
        ]
        # for name, param in reversed(list(bagua_module.named_parameters())):
        #     tensor = param.bagua_ensure_grad().to_bagua_tensor(name) # TODO: check duplicated names
        #     tensor_groups[0].append(tensor)
        return tensors

    def init_operations(
            self,
            bagua_module: BaguaModule,
            bucket: BaguaBucket,
    ):
        ...
        # bucket.backend_bucket.clear_ops()
        # if self.hierarchical_reduce:
        #     bucket.backend_bucket.append_centralized_synchronous_op(
        #         bagua_module.bagua_inter_node_communicator,
        #         bagua_module.bagua_intra_node_communicator,
        #         hierarchical=self.hierarchical_reduce,
        #         average=self.average,
        #     )
        # else:
        #     bucket.backend_bucket.append_centralized_synchronous_op(
        #         bagua_module.bagua_global_communicator,
        #         None,
        #         hierarchical=self.hierarchical_reduce,
        #         average=self.average,
        #     )
