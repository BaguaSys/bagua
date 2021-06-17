import torch
from ..distributed_module import DistributedModule
from bagua.torch_api.compression import Compressor


class DecentralizedReducer(DistributedModule):
    def __init__(
        self,
        module: torch.nn.Module,
        compressor: Compressor = Compressor.NoneCompressor,
        peer_selection_mode: str = "all",
        communication_interval: int = 1,
        **kwargs
    ):
        super(DecentralizedReducer, self).__init__(module)
        self.compressor = compressor
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval

    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
        return result

    def post_backward_fn(self, backend, **kwargs):
        torch.cuda.synchronize()
        backend.execute_post_backward_comm_ops()
        backend.wait_pending_post_backward_comm_ops()

    def set_communication_op(
        self,
        bagua_bucket,
        inter_communicator,
        intra_communicator=None,
        hierarchical_reduce=False,
        **kwargs
    ):
        bagua_bucket.append_decentralized_synchronous_op(
            inter_communicator,
            intra_communicator,
            hierarchical=hierarchical_reduce,
            compression=self.compressor.value,
            peer_selection_mode=self.peer_selection_mode,
            communication_interval=self.communication_interval,
        )
