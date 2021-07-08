import logging
import multiprocessing
import bagua_core as B
from bagua.service import AutotuneService
from collections import defaultdict
from . import env
from .env import (
    get_master_addr,
    get_world_size,
    get_rank,
    get_local_rank,
    get_local_size,
    get_default_bucket_size,
    get_bagua_service_port,
)
from .utils import flatten, unflatten, to_bagua_reduce_op
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from bagua.service.autotune_service import AutotuneClient
from functools import lru_cache


@lru_cache(maxsize=None)
def get_hyperparameters_service_client():
    hyperparameters_service_client = AutotuneClient(
        get_master_addr(), get_bagua_service_port()
    )
    return hyperparameters_service_client


@lru_cache(maxsize=None)
def get_backend(model_name: str):
    backend = B.BaguaCommBackendPy(100, device_id=get_local_rank())
    backend.device_id = get_local_rank()
    backend.stream = torch.cuda.Stream(priority=-1)
    backend.store = c10d._get_default_store()
    backend.internode_communicator = init_bagua_inter_communicator(
        model_name=model_name,
        stream=backend.stream,
        leader_rank=0,
        store=backend.store,
        device_id=backend.device_id,
    )
    backend.intranode_communicator = init_bagua_intra_communicator(
        model_name=model_name,
        stream=backend.stream,
        store=backend.store,
        device_id=backend.device_id,
    )
    backend.global_communicator = init_bagua_communicator(
        model_name=model_name,
        stream=backend.stream,
        store=backend.store,
        device_id=backend.device_id,
    )
    return backend


def run_flask_app():
    from flask import Flask

    autotune_service = AutotuneService(
        world_size=get_world_size(),
        autotune_level=env.get_autotune_level(),
        max_samples=env.get_autotune_max_samples(),
        sampling_confidence_time_s=env.get_autotune_sampling_confidence_time_s(),
        warmup_time_s=env.get_autotune_warmup_time_s(),
        is_output_autotune_log=env.get_is_output_autotune_log(),
        default_bucket_size=get_default_bucket_size(),
    )
    app = Flask(__name__)
    app = autotune_service.setup_app(app)
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    app.run(
        host="0.0.0.0",
        port=get_bagua_service_port(),
        debug=False,
    )


_autotune_server = None


def start_autotune_server():
    """Start autotune server in background."""
    global _autotune_server

    _autotune_server = multiprocessing.Process(target=run_flask_app)
    _autotune_server.daemon = True
    _autotune_server.start()


def init_process_group():
    """Initializes the PyTorch builtin distributed process group, and this will
    also initialize the distributed package, should be executed before all the
    APIs of bagua.

    Raises:
        RepeatedInitializationError: If you run this function repeatedly

    Examples::
        >>> import bagua.torch_api as bagua
        >>> bagua.init_process_group()
        >>> model = torch.nn.Sequential(
        ...    torch.nn.Linear(D_in, H),
        ...    torch.nn.ReLU(),
        ...    torch.nn.Linear(H, D_out),
        ...    )
        >>> optimizer = torch.optim.SGD(
        ...    model.parameters(),
        ...    lr=0.01,
        ...    momentum=0.9
        ...    )
        >>> model, optimizer = bagua_init(model, optimizer)
    """
    if not dist.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )  # fmt: off

    if get_rank() == 0 and _autotune_server is None:
        start_autotune_server()


def gen_nccl_unique_id(comm_type: str, root=0, store=None):
    key = f"{comm_type}-{root}-unique_id"

    if store is None:
        store = c10d._get_default_store()

    if get_rank() == root:
        idstr = B.BaguaSingleCommunicatorPy.generate_nccl_unique_id_str()
        store.set(key, idstr)
    else:
        idstr = store.get(key)
        idstr = str(idstr, encoding="utf-8")

    return idstr


def init_bagua_inter_communicator(
    model_name: str, stream, leader_rank=0, store=None, device_id=None
):
    if device_id is None:
        device_id = get_local_rank()
    nccl_unique_id = gen_nccl_unique_id(
        f"bagua_inter_comm_{model_name}", root=leader_rank, store=store
    )

    if get_rank() % get_local_size() != leader_rank:
        return None

    comm = B.BaguaSingleCommunicatorPy(
        rank=get_rank() // get_local_size(),
        nranks=get_world_size() // get_local_size(),
        device_id=device_id,
        stream_ptr=stream.cuda_stream,
        nccl_unique_id_str=nccl_unique_id,
    )
    comm.cuda_stream = stream
    logging.debug(
        "init bagua internode communicator ok, global rank: %s rank: %s",
        dist.get_rank(),
        comm.rank(),
    )
    return comm


def init_bagua_intra_communicator(model_name: str, stream, store=None, device_id=None):
    if device_id is None:
        device_id = get_local_rank()
    nccl_unique_id = gen_nccl_unique_id(
        f"bagua_intra_comm_{model_name}",
        root=get_rank() // get_local_size() * get_local_size(),
        store=store,
    )

    comm = B.BaguaSingleCommunicatorPy(
        rank=get_rank() % get_local_size(),
        nranks=get_local_size(),
        device_id=device_id,
        stream_ptr=stream.cuda_stream,
        nccl_unique_id_str=nccl_unique_id,
    )
    comm.cuda_stream = stream
    logging.debug(
        "init bagua intranode communicator ok, global rank: %s rank: %s",
        dist.get_rank(),
        comm.rank(),
    )
    return comm


def init_bagua_communicator(model_name: str, stream, store=None, device_id=None):
    if device_id is None:
        device_id = get_local_rank()
    nccl_unique_id = gen_nccl_unique_id(f"bagua_global_comm_{model_name}", store=store)

    comm = B.BaguaSingleCommunicatorPy(
        rank=get_rank(),
        nranks=get_world_size(),
        device_id=device_id,
        stream_ptr=stream.cuda_stream,
        nccl_unique_id_str=nccl_unique_id,
    )
    comm.cuda_stream = stream
    logging.debug(
        "init bagua global communicator ok, global rank: %s rank: %s",
        dist.get_rank(),
        comm.rank(),
    )
    return comm


def broadcast_coalesced(tensors, root=0, comm: B.BaguaSingleCommunicatorPy = None):
    for tensor in tensors:
        assert tensor.device != torch.device(
            "cpu"
        ), "input tensors must be CUDA and dense"

    if comm is None:
        comm = get_backend("").global_communicator

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        coalesced = flatten(tensors)
        comm.broadcast(coalesced.to_bagua_tensor().bagua_backend_tensor(), root)
        for buf, synced in zip(tensors, unflatten(coalesced, tensors)):
            buf.copy_(synced)

    # TODO: remove
    torch.cuda.synchronize()


def broadcast(tensor, root=0, comm: B.BaguaSingleCommunicatorPy = None):
    r"""Broadcasts the tensor to the whole communicator.

    `tensor` must have the same number of elements in all processes
    participating in the collective.

    Args:
        tensor (torch.Tensor): Data to be sent if `root` is the rank of
            current process, and tensor to be used to save received data
            otherwise.
        root (int, optional): Source rank. Defaults to 0.
        comm (B.BaguaSingleCommunicatorPy, optional): The bagua communicator
            to work on. If None, the global bagua communicator will be used.
            Defaults to None.
    """  # noqa: W293

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None:
        comm = get_backend("").global_communicator

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.broadcast(tensor.to_bagua_tensor().bagua_backend_tensor(), root)

    # TODO: remove
    torch.cuda.synchronize()


def reduce(tensor, dst, op=dist.ReduceOp.SUM, comm: B.BaguaSingleCommunicatorPy = None):
    r"""Reduces the tensor across all processes.

    Only the process whit rank `dst` is going to receive the final result.

    Args:
        tensor (torch.Tensor): Input and output of the collective. The
            function operates in-place.
        dst (int): Destination rank
        op (optional): one of the values from `torch.distributed.ReduceOp`
            enum. Specifies an operation used for element-wise reductions.
        comm (B.BaguaSingleCommunicatorPy, optional): The bagua communicator to
            work on. If None the global bagua communicator will be used.
            Defaults to None.
    """  # noqa: W293

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None:
        comm = get_backend("").global_communicator

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.reduce(
            tensor.to_bagua_tensor().bagua_backend_tensor(), dst, to_bagua_reduce_op(op)
        )

    torch.cuda.synchronize()


def allreduce_coalesced(
    tensors,
    op=dist.ReduceOp.SUM,
    comm: B.BaguaSingleCommunicatorPy = None,
):
    for tensor in tensors:
        assert tensor.device != torch.device(
            "cpu"
        ), "input tensors must be CUDA and dense"

    if comm is None:
        comm = get_backend("").global_communicator

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        coalesced = flatten(tensors)
        comm.allreduce(
            coalesced.to_bagua_tensor("allreduce_coalesced"), to_bagua_reduce_op(op)
        )

        for buf, synced in zip(tensors, unflatten(coalesced, tensors)):
            buf.copy_(synced)

    # TODO: remove
    torch.cuda.synchronize()


def allreduce(
    tensor,
    op=dist.ReduceOp.SUM,
    comm: B.BaguaSingleCommunicatorPy = None,
):
    """Reduces the tensor data across all machines in such a way that all get
    the final result. After the call tensor is going to be bitwise identical
    in all processes.

    Args:
        tensor (torch.Tensor): Input and output of the collective. The
            function operates in-place.
        op (optional): one of the values from `torch.distributed.ReduceOp` enum. Specifies an operation used for element-wise reductions.
        comm (B.BaguaSingleCommunicatorPy, optional): The bagua communicator to
            work on. If None the global bagua communicator will be used.
            Defaults to None.

    Examples:
        >>> from bagua.torch_api import allreduce
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> allreduce(tensor)
        >>> tensor
        tensor([4, 6]) # Rank 0
        tensor([4, 6]) # Rank 1

        >>> # All tensors below are of torch.cfloat type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * rank * (1+1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j]) # Rank 0
        tensor([3.+3.j, 4.+4.j]) # Rank 1
        >>> allreduce(tensor)
        >>> tensor
        tensor([4.+4.j, 6.+6.j]) # Rank 0
        tensor([4.+4.j, 6.+6.j]) # Rank 1
    """  # noqa: E501

    assert tensor.device != torch.device("cpu"), "input tensor must be CUDA and dense"

    if comm is None:
        comm = get_backend("").global_communicator

    event = torch.cuda.current_stream().record_event()
    comm.cuda_stream.wait_event(event)

    with torch.cuda.stream(comm.cuda_stream):
        comm.allreduce(
            tensor.to_bagua_tensor().bagua_backend_tensor(), to_bagua_reduce_op(op)
        )

    # TODO: remove
    torch.cuda.synchronize()
