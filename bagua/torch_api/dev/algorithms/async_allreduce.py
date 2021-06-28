#!/usr/bin/env python3

from bagua.torch_api.dev.bucket import BaguaBucket
from bagua.torch_api.dev.distributed_dev import BaguaModule
from bagua.torch_api.dev.algorithms import Algorithm

class AsyncAllReduceAlgorithm(Algorithm):
    def __init__(self, sync_interval_ms: int):
        """
        Create an instance of the
        `AsyncAllReduce <https://baguasys.github.io/tutorials/algorithms/async-allreduce.html>`_
        algorithm.

        Args:
            sync_interval_ms (int): How many milliseconds between two model synchronizations.
        """
        self.sync_interval_ms = sync_interval_ms

    def init_tensors(self, bagua_module: BaguaModule) -> List[List[BaguaTensor]]:
        parameters = bagua_module._bagua_build_params()
        tensor_groups = [[
           param.to_bagua_tensor(name)
            for name, param in parameters.__reversed__()
        ]]
        # # TODO: consider optimizer groups
        # for name, param in reversed(list(bagua_module.named_parameters())):
        #     tensor = param.bagua_ensure_grad().to_bagua_tensor(name) # TODO: check duplicated names
        #     tensor_groups[0].append(tensor)
        return tensor_groups

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
