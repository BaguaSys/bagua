import torch
from ..distributed_module import DistributedModule
from bagua.torch_api.distributed_define import ReduceOp
from bagua.torch_api.compression import Compressor


class ScatterGatherAllreducer(DistributedModule):
    def __init__(
        self,
        module: torch.nn.Module,
        reduce_op: ReduceOp = ReduceOp.Average,
        compressor: Compressor = Compressor.NoneCompressor,
        **kwargs
    ):

        super(ScatterGatherAllreducer, self).__init__(module)
        self.reduce_op = reduce_op
        self.compressor = compressor

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
            scattergather=True,
            compression=self.compressor.value,
        )
