import enum
import torch
from torch.autograd import Function
import torch.distributed as dist


# must be consistent with Aluminum ReductionOperator: https://github.com/BaguaSys/Aluminum/blob/master/include/aluminum/base.hpp
class ReduceOp(enum.IntEnum):
    """An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``, ``MIN``, ``MAX``, ``BAND``,
    ``BOR``, ``BXOR`` and ``AVG``."""

    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BOR = 7
    BAND = 8
    BXOR = 9
    AVG = 10


def torch_reduce_op_to_bagua(op):
    if op is torch.distributed.ReduceOp.SUM:
        return ReduceOp.SUM
    elif op is torch.distributed.ReduceOp.MAX:
        return ReduceOp.MAX
    else:
        raise Exception("Unexpect input={}".format(op))


def all_reduce(tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call the returned tensor is going to be bitwise
    identical in all processes.

    Arguments:
        tensor (Tensor): Input of the collective.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective

    """
    if group is None:
        group = dist.group.WORLD

    return _AllReduce.apply(op, group, tensor)


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone()
        comm = group.bagua_patch().bagua_get_global_communicator()

        event = torch.cuda.current_stream().record_event()
        comm.cuda_stream.wait_event(event)

        with torch.cuda.stream(comm.cuda_stream):
            comm.allreduce_inplace(
                tensor.to_bagua_tensor().bagua_backend_tensor(),
                int(torch_reduce_op_to_bagua(op)),
            )

        comm.cuda_stream.synchronize()

        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)
