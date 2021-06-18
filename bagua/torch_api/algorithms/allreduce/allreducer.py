import torch
from ..distributed_module import DistributedModule
from bagua.torch_api.distributed_define import ReduceOp


class Allreducer(DistributedModule):
    def __init__(
        self, module: torch.nn.Module, reduce_op: ReduceOp = ReduceOp.Average, **kwargs
    ):
        super(Allreducer, self).__init__(module)
        self.reduce_op = reduce_op

    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
        return result

    def set_communication_op(
        self,
        bagua_bucket,
        inter_communicator,
        intra_communicator=None,
        hierarchical_reduce=False,
        **kwargs
    ):
        bagua_bucket.append_centralized_synchronous_op(
            inter_communicator,
            intra_communicator,
            hierarchical=hierarchical_reduce,
            average=(self.reduce_op == ReduceOp.Average),
        )
