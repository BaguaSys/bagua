import copy
import time
import torch
from torch.autograd import Variable
import logging
from typing import Optional, List, Dict, Union
import traceback
from .distributed_define import BucketType
from .communication import (
    is_initialized,
    allreduce,
    broadcast,
    _get_global_state,
    get_bagua_hyperparameters,
    get_hyperparameters_service_client,
)
from .env import (
    get_rank,
    get_world_size,
    get_local_size,
    get_autotune_level,
)
from .utils import (
    to_bagua_datatype,
    flatten_module_params,
    average_by_removing_extreme_values,
)
from .exceptions import UnsupportedAlgorithmException
from .algorithms.decentralize import DecentralizedReducer
from .algorithms.allreduce import Allreducer, ScatterGatherAllreducer
from .compression import Compressor
from bagua.bagua_define import (
    TensorDtype,
    TensorDeclaration,
    DistributedAlgorithm,
    BaguaHyperparameter,
)
import bagua_core as B


class DistributedModule(torch.nn.Module):
    r"""
    A base class for distributed module.
    """

    def __init__(
        self,
        module: torch.nn.Module,
    ):
        super(DistributedModule, self).__init__()
        self.module = module
        if hasattr(module, "_bagua_params_and_buffers_to_ignore"):
            ignore_para = module._bagua_params_and_buffers_to_ignore
            self.parameters_to_ignore = [("module." + k) for k in ignore_para]
        else:
            self.parameters_to_ignore = []

    def unwrap(self):
        r"""
        Return the unwraped module.
        """

        return self.module

    def forward(self, *inputs, **kwargs):
        r"""
        Execute the forward process and return the output.
        """

        result = self.module(*inputs, **kwargs)
        return result


class Reducer(object):
    r"""In order to improve communication efficiency, the distributed
    algorithm chunks parameters into many buckets. A bucket is the
    minimum unit of communication between devices in bagua.
    This module is the bucket manager, providing bucket operation methods.

    The process mainly consists the following two cases:

        1. bucket_initialized is False:
            1.1 add_param

            1.2 initialize_buckets -> register_models

            1.3 mark_bucket_ready

            1.4 mark_on_complete

        2. bucket_initialized is True:
            2.1 mark_tensor_ready

            2.2 mark_on_complete

    Args:
        module (DistributedModule): Module to be parallelized.
        optimizers (torch.optim.Optimizer or list of torch.optim.Optimizer):
            Optimizer(s) for the module. It can contain one or more
            PyTorch optimizers.
        bucket_type (BucketType): Type of elements in a communication bucket,
            could be either module parameters, weights or gradients.
        hierarchical_reduce (bool): Enable hierarchical reduce, which will
            perform an intra-node allreduce, followed by an inter-node reduce
            defined by different `module`, and an intra-node broadcast
            at the end.
        align_bytes (bool): Number to bytes to be aligned for each
            communication bucket.
        chunking (bool): For alltoall communication pattern,
            set `chunking` to `True`.
        fusion (bool): To reset parameter data pointer so that they can use
            faster code paths, set `fusion` to `True`.
        decentralize_reduce (bool): Whether execute the decentralize
            communication. Default: `False`.
        buckets (List[List[TensorDeclaration]]): Parameter buckets.

    """

    def __init__(
        self,
        module: DistributedModule,
        optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        bucket_type: BucketType,
        hierarchical_reduce: bool,
        align_bytes: int,
        chunking: bool,
        fusion: bool,
        decentralize_reduce: bool = False,
        buckets: List[List[TensorDeclaration]] = [],
        **kwargs,
    ):
        super(Reducer, self).__init__()

        self.module = module

        self.optimizers = optimizers
        assert isinstance(
            self.optimizers, (list, tuple)
        ), "Invalid optimizers type({}), should be list".format(type(optimizers))
        self.client = get_hyperparameters_service_client()

        self.bucket_type = bucket_type

        self.hierarchical_reduce = hierarchical_reduce
        self.decentralize_reduce = decentralize_reduce

        if not hasattr(module, "set_communication_op"):
            raise ValueError("set_communication_op is not implemented")

        self.post_backward_fn = None
        if hasattr(module, "post_backward_fn"):
            self.post_backward_fn = module.post_backward_fn

        self.bucket_initialized = False

        # temporary for sync bucket structures
        self.param_list = []
        for name, param in self.module.named_parameters():
            if param.requires_grad and name not in self.module.parameters_to_ignore:
                self.param_list.append(param)
            else:
                logging.debug(f"skip param: {name}")

        self.param_i = {
            id(param): i for i, param in enumerate(self.param_list)
        }  # every worker maintains this id -> params correspondence, we do not need id(param) to be the same on every worker
        self.param_name = {
            id(param): name for name, param in self.module.named_parameters()
        }
        self.param_dict = {self.param_name[id(p)]: p for p in self.param_list}

        # bucket structures
        self.buckets: List[List[TensorDeclaration]] = copy.deepcopy(buckets)

        priority = kwargs.get("priority")
        if priority is not None:
            logging.debug("set stream priority to {}".format(priority))
            self.priority = priority
        else:
            self.priority = -1

        self.tensor_events = [
            torch.cuda.Event(enable_timing=False, blocking=False)
            for _ in self.param_list
        ]

        if align_bytes < 8:
            raise ValueError("align bytes must be larger than 8")

        if align_bytes & (align_bytes - 1):
            raise ValueError("align bytes should be a power of 2")

        self.align_bytes = align_bytes
        self.chunking = chunking

        self.current_device = torch.cuda.current_device()
        self.current_stream = torch.cuda.current_stream()

        if self.hierarchical_reduce:
            self.bagua_communicator = _get_global_state().get_internode_communicator()
            self.bagua_intra_communicator = (
                _get_global_state().get_intranode_communicator()
            )
            comm_nchunks = get_world_size() // get_local_size()
            logging.info(
                f"hierarchical reduce is enabled, intranode comm size: {self.bagua_intra_communicator.nranks()}"
            )
        else:
            self.bagua_communicator = _get_global_state().get_global_communicator()
            self.bagua_intra_communicator = None
            comm_nchunks = get_world_size()
            logging.info("hierarchical reduce is disabled")

        self.bagua_backend = _get_global_state().get_backend()

        if self.chunking and comm_nchunks * 32 > self.align_bytes:
            self.align_bytes = comm_nchunks * 32
            logging.debug(f"changing align bytes to {self.align_bytes} for chunking")

        self.fusion = fusion
        self.step_counter = 0

        self.tensor_list: List[TensorDeclaration] = []
        self.param_buckets: List[List[torch.Tensor]] = []
        self.bagua_tensor: Dict[str, B.BaguaTensorPy] = {}

    def fill_slot(self, param):
        r"""
        Get the value of parameters.
        """
        if self.bucket_type == BucketType.Gradient:
            return param.grad.data
        elif self.bucket_type == BucketType.Weight:
            return param.data
        elif self.bucket_type == BucketType.Param:
            return param

    def initialize_buckets(self) -> List[List[torch.Tensor]]:
        r"""
        Initialize parameter buckets.

        .. note:: Initialize_buckets MUST execute after the first round
            of backward.

        Returns:
            parameter buckets.
        """
        if len(self.buckets) == 0:
            param_group_info = {}
            param_groups = [
                group
                for optimizer in self.optimizers
                for group in optimizer.param_groups
            ]
            for i, group in enumerate(param_groups):
                for param in group["params"]:
                    param_group_info[self.param_name[id(param)]] = i

            whether_to_bucket = True
            if self.decentralize_reduce:
                self.tensor_list = [
                    TensorDeclaration(
                        {
                            "name": self.param_name[id(param)],
                            "num_elements": param.numel(),
                            "dtype": to_bagua_datatype(param.dtype),
                        }
                    )
                    for param in self.param_list
                ]
                whether_to_bucket = False

            rsp = self.client.register_models(
                self.tensor_list, param_group_info, whether_to_bucket=whether_to_bucket
            )
            hp = BaguaHyperparameter(**rsp.json()["recommended_hyperparameters"])
            self.buckets = hp.buckets
            get_bagua_hyperparameters().buckets = self.buckets

        logging.debug("Initialized bucktes={}".format(self.buckets))
        self.param_buckets = []
        for bucket in self.buckets:
            self.param_buckets.append([self.param_dict[td["name"]] for td in bucket])

        if self.fusion:
            dtype_list = [
                TensorDtype.F32.value,
                TensorDtype.F16.value,
                TensorDtype.U8.value,
            ]
            for dtype in dtype_list:
                buckets = [
                    bucket
                    for bucket in self.param_buckets
                    if to_bagua_datatype(bucket[0].dtype) == dtype
                ]
                flatten_module_params(buckets, self.align_bytes)
                torch.cuda.empty_cache()

        self.register_bagua_buckets()
        self.bucket_initialized = True

        return self.param_buckets

    def register_bagua_buckets(self):
        r"""
        Register bagua buckets.
        """

        def new_bagua_tensor(param):
            p = self.fill_slot(param)
            bagua_tensor = B.BaguaTensorPy(
                ptr=p.data_ptr(),
                num_elem=param.numel(),
                num_elem_allocated=param.__dict__.get("allocated_size", param.numel()),
                dtype=to_bagua_datatype(param.dtype),
                device_id=p.device.index,
            )
            param.bagua_tensor = bagua_tensor
            param_name = self.param_name[id(param)]
            self.bagua_tensor[param_name] = bagua_tensor
            return bagua_tensor

        # register bagua tensor and bagua bucket
        bagua_buckets = []
        for index, bucket_params in enumerate(self.param_buckets):
            bagua_tensors = [new_bagua_tensor(p) for p in bucket_params]
            # Note: align_bytes does not work when inplace=True
            bagua_bucket = B.BaguaBucketPy(
                "bucket_" + str(index),
                bagua_tensors,
                inplace=self.fusion,
                align_bytes=self.align_bytes,
            )

            kwargs = {"param_name": self.param_name}
            self.module.set_communication_op(
                bagua_bucket,
                inter_communicator=self.bagua_communicator,
                intra_communicator=self.bagua_intra_communicator,
                hierarchical_reduce=self.hierarchical_reduce,
                **kwargs,
            )

            bagua_buckets.append(bagua_bucket)

        self.bagua_backend.register_ordered_buckets(bagua_buckets)

    def add_param(self, param):
        r"""
        Add parameter into tensor_list.
        """
        if id(param) not in self.param_i:
            return

        data = self.fill_slot(param)
        self.tensor_list.append(
            {
                "name": self.param_name[id(param)],
                "num_elements": param.numel(),
                "dtype": to_bagua_datatype(param.dtype),
            }
        )

    def mark_bucket_ready(self, bucket, bucket_idx):
        r"""
        Mark all tensors in the bucket ready.
        """
        for param in bucket:
            self.mark_tensor_ready(param)

    def mark_tensor_ready(self, param):
        r"""
        Mark the tensor ready when got its gradient.
        """
        param_name = self.param_name[id(param)]
        if param_name not in self.bagua_tensor:  # no bagua_tensor no mark ready
            return

        if not self.fusion:
            # reset bagua tensor pointer
            p = self.fill_slot(param)
            self.bagua_tensor[param_name].reset_ptr(p.data_ptr())

        ready_event = self.tensor_events[self.param_i[id(param)]]
        self.current_stream.record_event(ready_event)

        self.bagua_backend.mark_communication_ready(
            self.bagua_tensor[param_name], ready_event.cuda_event
        )

    def mark_on_complete(self):
        r"""
        Mark all buckets have finished thier reduce process.
        """
        self.bagua_backend.wait_pending_comm_ops()

        if self.post_backward_fn is not None:
            self.post_backward_fn(self.bagua_backend)

        self.step_counter += 1


class OverlappingWrapper(torch.nn.Module):
    r"""
    This class defines the process of communication-computation overlap.

    Arguments:
        module (torch.nn.Module): A distributed module to be overlapped.
        optimizers (torch.optim.Optimizer or list of torch.optim.Optimizer):
            Optimizer(s) for the module. It can contain one or more
            PyTorch optimizers.
        delay_reduce (bool): Delay all communication to the end of the
            backward pass. This disables overlapping communication with
            computation. Default value is `False`.
        bucket_type (BucketType): Type of elements in a communication bucket,
            could be either module parameters, weights or gradients.
        hierarchical_reduce (bool): Enable hierarchical reduce, which will
            perform an intra-node allreduce, followed by an inter-node reduce
            defined by different `module`, and an intra-node broadcast
            at the end.
        decentralize_reduce (bool): For decentralize training, set
            `decentralize_reduce` to `True`.
        align_bytes (int): Number to bytes to be aligned for each
            communication bucket.
        chunking (bool): For alltoall communication pattern,
            set `chunking` to `True`.
        fusion (bool): To reset parameter data pointer so that they can use
            faster code paths, set `fusion` to `True`.

    .. note::
        This implementation benefits a lot from `apex.parallel.DistributedDataParallel`.

    """

    def __init__(
        self,
        module: DistributedModule,
        optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        delay_reduce: bool = False,
        bucket_type=BucketType.Gradient,
        hierarchical_reduce: bool = False,
        decentralize_reduce: bool = False,
        parameter_manager=None,
        align_bytes: int = 8,
        chunking: bool = False,
        fusion: bool = True,
        **kwargs,
    ):
        super(OverlappingWrapper, self).__init__()

        self.hook_list: List = []
        self.module = module
        self.optimizers = optimizers
        self.delay_reduce = delay_reduce
        self.compute_communication_overlap = not self.delay_reduce
        self.bucket_type = bucket_type
        self.hierarchical_reduce = hierarchical_reduce
        self.decentralize_reduce = decentralize_reduce
        self.align_bytes = align_bytes
        self.chunking = chunking
        self.fusion = fusion
        self.kwargs = kwargs
        self.align_bytes = align_bytes
        self.chunking = chunking

        self.reset_reducer()

    def reset_reducer(
        self,
        hierarchical_reduce: Optional[bool] = None,
        buckets: Optional[List[List[TensorDeclaration]]] = None,
    ):
        r"""
        Reset the parameter reducer.

        Arguments:
            hierarchical_reduce (bool): Enable hierarchical reduce.
            buckets (List[List[TensorDeclaration]]): Parameter buckets.
        """
        for h in self.hook_list:
            h.remove()

        if hierarchical_reduce is not None:
            self.hierarchical_reduce = hierarchical_reduce
        if buckets is not None:
            self.kwargs["buckets"] = buckets
        self.kwargs["decentralize_reduce"] = self.decentralize_reduce
        self.reducer = Reducer(
            self.module,
            self.optimizers,
            self.bucket_type,
            self.hierarchical_reduce,
            self.align_bytes,
            self.chunking,
            self.fusion,
            **self.kwargs,
        )
        self.create_hooks()

    def create_hooks(self):
        r"""
        Defines a number of hooks used to reduce communication buckets
        in backward process.
        """

        self.grad_accs = []
        self.decen_callonce_flag = False
        for param in self.module.parameters():
            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]

                def make_hook(param):
                    def reduce_fallback(skip_reduce=False):
                        if skip_reduce:
                            logging.debug("skip reduce")
                            return

                        for i, bucket in enumerate(self.reducer.param_buckets):
                            self.reducer.mark_bucket_ready(bucket, i)

                        self.reducer.mark_on_complete()

                    def synchronize():
                        self.reducer.initialize_buckets()
                        reduce_fallback()

                    def register_post_backward_func(callback_func):
                        """
                        Queue callback_func to the execution engine
                        """

                        def traceback_callback_func():
                            try:
                                callback_func()
                            except:
                                print(traceback.format_exc())
                                logging.error(traceback.format_exc())
                                raise

                        if not self.callback_queued:
                            Variable._execution_engine.queue_callback(
                                traceback_callback_func
                            )
                            self.callback_queued = True

                    def _decentralize_hook(*unused):
                        if self.decen_callonce_flag:
                            return

                        self.decen_callonce_flag = True
                        if not self.reducer.bucket_initialized:
                            # decentralize algorithm skip communications at the first step
                            register_post_backward_func(
                                lambda: self.reducer.initialize_buckets()
                            )
                            return

                        if self.compute_communication_overlap:
                            # initiate communications only once at each iteration
                            for i, bucket in enumerate(self.reducer.param_buckets):
                                self.reducer.mark_bucket_ready(bucket, i)

                            register_post_backward_func(self.reducer.mark_on_complete)
                        else:
                            register_post_backward_func(reduce_fallback)

                    def _hook(*unused):
                        if not self.reducer.bucket_initialized:
                            self.reducer.add_param(param)
                            register_post_backward_func(synchronize)
                            return

                        if (
                            self.compute_communication_overlap
                        ):  # overlap reduce and backward
                            self.reducer.mark_tensor_ready(param)
                            register_post_backward_func(self.reducer.mark_on_complete)
                        else:
                            register_post_backward_func(reduce_fallback)

                    return _decentralize_hook if self.decentralize_reduce else _hook

                h = grad_acc.register_hook(make_hook(param))

                self.hook_list.append(h)
                self.grad_accs.append(grad_acc)

    def forward(self, *inputs, **kwargs):
        r"""
        Overwrite the forward process for a distributed module with
        communication-computation overlap.
        """
        result = self.module(*inputs, **kwargs)
        self.callback_queued = False
        self.decen_callonce_flag = False
        return result


class ModelSwitchWrapper(torch.nn.Module):
    r"""
    `ModelSwitchWrapper` is designed to switch distributed algorithms during
    training process. It mainly has two functions.
    The first is transform the original module to a distributed module.
    Second, this class can change the distributed mode to another one
    in the training process.
    Args:
        module (torch.nn.Module): Network definition to be run
            in multi-gpu/distributed mode.
        optimizer (torch.optim.Optimizer or list of torch.optim.Optimizer):
            Optimizer(s) for the module. It can contain one
            or more PyTorch optimizers.
        broadcast_buffers (bool): Flag that enables syncing (broadcasting)
            buffers of the module at **the first iteration** of the forward
            function. Default: `True`.
        delay_reduce (bool): Overlap communication with computation.
            Default: `True`.
        hierarchical_reduce (bool): Enable hierarchical reduce. For
            `GradientAllReduce` algorithm, default value is `False`,
            otherwise, default value is `True`.
        message_size (int): Minimum bytes in a communication bucket.
            Default: `10_000_000`.
        intra_comm_root_rank (int): Root rank of intra communication.
            Default: `0`.

    Returns:
        Distributed module.

    Examples::

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
        >>> model = ModelSwitchWrapper(
        ...    model = model,
        ...    optimizer = optimizer,
        ...    broadcast_buffers = broadcast_buffers,
        ...    delay_reduce = delay_reduce,
        ...    hierarchical_reduce = hierarchical_reduce,
        ...    message_size = message_size,
        ...    **kwargs,
        ...    ).switch_to(DistributedAlgorithm.GradientAllReduce)
        >>> train A epochs
        >>> model.switch_to(DistributedAlgorithm.Decentralize)
        >>> train B epochs
        >>> model.switch_to(DistributedAlgorithm. GradientAllReduce)
        >>> continue training
        >>> ...
    """

    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        broadcast_buffers: bool = True,
        delay_reduce: bool = False,
        hierarchical_reduce: Union[bool, None] = None,
        message_size: int = 10_000_000,
        intra_comm_root_rank: int = 0,
        **kwargs,
    ):
        super(ModelSwitchWrapper, self).__init__()

        self.module = module
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizers = [optimizer]
        else:
            optimizers = optimizer

        self.optimizers = optimizers
        self.broadcast_buffers = broadcast_buffers
        self.delay_reduce = delay_reduce
        self.hierarchical_reduce = hierarchical_reduce
        self.message_size = message_size
        self.intra_comm_root_rank = intra_comm_root_rank
        self.kwargs: dict = kwargs

        self.bagua_module: torch.nn.Module = torch.nn.Module()
        self.stage = 0
        self.autotune_client = get_hyperparameters_service_client()
        self.step_counter = 0

        self.last_hit_tp = time.time()
        self.score_record_list: List[float] = []
        self.autotune_flag = get_autotune_level() >= 1

    def switch_to(
        self,
        distributed_algorithm: Union[DistributedAlgorithm, str],
    ):
        r"""
        Switch the initial module to distributed module.

        Arguments:
            distributed_algorithm (DistributedAlgorithm): Distributed
                algorithm used to average gradients or weights across
                all workers. Default: `DistributedAlgorithm.GradientAllReduce`.

        Returns:
            Return the distributed module to cover the initial one.
        """
        # Reset Autotune Server
        self.autotune_client.reset()
        self.bagua_module = self.module
        if isinstance(distributed_algorithm, str):
            distributed_algorithm = DistributedAlgorithm.from_str(distributed_algorithm)

        # sync params at the start of each training stage
        if self.stage == 0:
            broadcast_parameters(self.bagua_module, self.broadcast_buffers)
        else:
            allreduce_parameters(self.bagua_module)

        self.stage += 1
        # Initialize distributed module reducer
        if distributed_algorithm == DistributedAlgorithm.GradientAllReduce:
            self.bagua_module = Allreducer(self.bagua_module, **self.kwargs)
            self.bagua_module = OverlappingWrapper(
                self.bagua_module,
                self.optimizers,
                delay_reduce=self.delay_reduce,
                hierarchical_reduce=(
                    False
                    if self.hierarchical_reduce is None
                    else self.hierarchical_reduce
                ),
                message_size=self.message_size,
                chunking=False,
                **self.kwargs,
            )
        elif distributed_algorithm == DistributedAlgorithm.ScatterGatherAllReduce:
            self.bagua_module = ScatterGatherAllreducer(
                self.bagua_module, **self.kwargs
            )
            self.bagua_module = OverlappingWrapper(
                self.bagua_module,
                self.optimizers,
                delay_reduce=self.delay_reduce,
                hierarchical_reduce=(
                    True
                    if self.hierarchical_reduce is None
                    else self.hierarchical_reduce
                ),
                message_size=self.message_size,
                chunking=True,
                **self.kwargs,
            )
        elif distributed_algorithm == DistributedAlgorithm.Decentralize:
            self.bagua_module = DecentralizedReducer(self.bagua_module, **self.kwargs)
            self.bagua_module = OverlappingWrapper(
                self.bagua_module,
                self.optimizers,
                delay_reduce=self.delay_reduce,
                bucket_type=BucketType.Weight,
                message_size=self.message_size,
                hierarchical_reduce=(
                    True
                    if self.hierarchical_reduce is None
                    else self.hierarchical_reduce
                ),
                decentralize_reduce=True,
                chunking=False,
                **self.kwargs,
            )
        elif distributed_algorithm == DistributedAlgorithm.QuantizeAllReduce:
            self.bagua_module = ScatterGatherAllreducer(
                self.bagua_module, compressor=Compressor.Uint8Compressor, **self.kwargs
            )
            self.bagua_module = OverlappingWrapper(
                self.bagua_module,
                self.optimizers,
                delay_reduce=self.delay_reduce,
                hierarchical_reduce=(
                    True
                    if self.hierarchical_reduce is None
                    else self.hierarchical_reduce
                ),
                message_size=self.message_size,
                chunking=True,
                **self.kwargs,
            )
        else:
            raise UnsupportedAlgorithmException(distributed_algorithm)

        get_bagua_hyperparameters().update(
            {
                "distributed_algorithm": distributed_algorithm.value,  # type: ignore
                "is_hierarchical_reduce": bool(self.hierarchical_reduce),
            }
        )
        # TODO: Configure the default hyperparameters for different algorithms

        return self

    def state_dict(self, **kwargs):
        r"""
        Fetch the module's state_dict.
        """
        return self.module.state_dict(**kwargs)

    def report_metrics(self, score_record_list):
        r"""
        Logging the metrics of auto_tune algorithm.
        """
        if len(score_record_list) == 0:
            iter_per_seconds = 0.0
        else:
            iter_per_seconds = sum(score_record_list) / len(score_record_list)
        logging.info("score_record_list={}".format(score_record_list))
        denoised_iter_per_seconds, std, _ = average_by_removing_extreme_values(
            score_record_list
        )
        logging.info(
            "iter_per_seconds={}, denoised_iter_per_seconds={}, std={}".format(
                iter_per_seconds,
                denoised_iter_per_seconds,
                std,
            )
        )

        self.autotune_client.report_metrics(
            rank=get_rank(),
            unix_timestamp=time.time(),
            train_iter=self.step_counter,
            iter_per_seconds=iter_per_seconds,
            denoised_iter_per_seconds=denoised_iter_per_seconds,
            hyperparameters=get_bagua_hyperparameters().dict(),
        )

    def ask_and_update_hyperparameters(self) -> bool:
        r"""
        Execute the environment search process by auto_tune
        and update the hyper-parameters.
        """
        rsp = self.autotune_client.ask_hyperparameters(
            rank=get_rank(), train_iter=self.step_counter
        )
        recommended_hyperparameters = copy.deepcopy(get_bagua_hyperparameters()).update(
            rsp.json()["recommended_hyperparameters"]
        )
        self.autotune_flag = rsp.json()["is_autotune_processing"]

        def update_hyperparameters(rec_hp):
            my_hp = get_bagua_hyperparameters()
            logging.info(
                "rec_hp={}, my_hp={}, rec_hp.buckets={}".format(
                    rec_hp.dict(), my_hp.dict(), rec_hp.buckets
                )
            )
            if rec_hp.distributed_algorithm != my_hp.distributed_algorithm:
                # The proposal to replace the communication algorithm is currently not supported
                return False

            if (
                DistributedAlgorithm(rec_hp.distributed_algorithm)
                is DistributedAlgorithm.GradientAllReduce
            ):
                rec_kwargs = {}
                if rec_hp.buckets and rec_hp.buckets != my_hp.buckets:
                    rec_kwargs["buckets"] = rec_hp.buckets
                if rec_hp.is_hierarchical_reduce != my_hp.is_hierarchical_reduce:
                    rec_kwargs["hierarchical_reduce"] = rec_hp.is_hierarchical_reduce

                if rec_kwargs:
                    logging.info("update hyperparameters to {}".format(rec_kwargs))
                    self.bagua_module.reset_reducer(**rec_kwargs)
                    get_bagua_hyperparameters().update(rec_hp.dict())
            elif (
                DistributedAlgorithm(rec_hp.distributed_algorithm)
                is DistributedAlgorithm.Decentralize
            ):
                # TODO: support decentralize autotune
                return False

            return True

        return update_hyperparameters(recommended_hyperparameters)

    def forward(self, *inputs, **kwargs):
        r"""
        Overwrite the forward processs and return the output.
        """
        assert self.bagua_module is not None
        result = self.bagua_module(*inputs, **kwargs)

        if self.module.training:
            cycle_step = 100
            if self.step_counter != 0 and self.step_counter % cycle_step == 0:
                st = time.time()

                if get_autotune_level() >= 1 and self.autotune_flag:
                    self.score_record_list.append(
                        cycle_step / float(time.time() - self.last_hit_tp)
                    )

                    self.report_metrics(self.score_record_list)
                    whether_the_parameter_updated = (
                        self.ask_and_update_hyperparameters()
                    )
                    if whether_the_parameter_updated:
                        self.score_record_list.clear()

                    self.last_hit_tp = time.time()

                logging.info("autotune overhead={}".format(time.time() - st))

            self.step_counter += 1

        return result


def _get_module_params_and_buffers(module, broadcast_buffers=True):
    r"""
    Get the module parameters (and buffers).
    Returns:
        module's parameters (and buffers).
    """
    if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
        parameters_to_ignore = module._ddp_params_and_buffers_to_ignore
    else:
        parameters_to_ignore = []

    module_states = []
    # Get the module parameters and buffers.
    if broadcast_buffers is True:
        for name, param in module.state_dict().items():
            if name not in parameters_to_ignore:
                module_states.append(param)
    # Only get the module parameters.
    elif broadcast_buffers is False:
        for name, param in module.named_parameters():
            if name not in parameters_to_ignore:
                module_states.append(param)
    return module_states


def broadcast_parameters(module, broadcast_buffers=True):
    r"""
    Broadcast the parameters (and buffers) for synchronization in the
    beginning. If `broadcast_buffers` is `False`, the buffers won't be
    synchronized (broadcasted) in the beginning.
    """
    from .communication import _get_global_state

    module_states = _get_module_params_and_buffers(
        module, broadcast_buffers=broadcast_buffers
    )

    authoritative_rank = 0
    for state in module_states:
        broadcast(state, root=authoritative_rank)


def allreduce_parameters(module):
    r"""
    Allreduce the parameters and buffers for synchronization
    at each time of switching distributed algorithms.
    """
    from .communication import _get_global_state

    module_states = _get_module_params_and_buffers(module)

    for state in module_states:
        allreduce(state, average=True)


def bagua_init(
    module: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    distributed_algorithm: Union[
        DistributedAlgorithm, str
    ] = DistributedAlgorithm.GradientAllReduce,
    broadcast_buffers: bool = True,
    delay_reduce: bool = False,
    hierarchical_reduce: Union[bool, None] = None,
    message_size: int = 10_000_000,
    **kwargs,
):
    r"""`bagua_init` is a module wrapper that enables easy multiprocess
    distributed data parallel training using different distributed algorithms.

    Arguments:
        module (torch.nn.Module): Network definition to be run
            in multi-gpu/distributed mode.
        optimizer (torch.optim.Optimizer or list of torch.optim.Optimizer):
            Optimizer(s) for the module. It can contain one
            or more PyTorch optimizers.
        distributed_algorithm (DistributedAlgorithm): Distributed algorithm
            used to average gradients or weights across all workers.
            Default: `DistributedAlgorithm.GradientAllReduce`.
        broadcast_buffers (bool): Flag that enables syncing (broadcasting)
            buffers of the module at **the first iteration** of the forward
            function. Default: `True`.
        delay_reduce (bool): Delay all communication to the end of the
            backward pass. This disables overlapping communication with
            computation. Default value is `False`.
        hierarchical_reduce (bool): Enable hierarchical reduce. For
            `GradientAllReduce` algorithm, default value is `False`,
            otherwise, default value is `True`.
        message_size (int): Minimum bytes in a communication bucket.
            Default: `10_000_000`.

    Returns:
        Distributed module.

    Examples::

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
        >>> model, optimizer = bagua_init(
        ...    model,
        ...    optimizer,
        ...    broadcast_buffers=True
        ...    )
    """

    assert is_initialized(), "Must call bagua.init_process_group() first!"

    # Set DistributedAlgorithm Wrapper
    module = ModelSwitchWrapper(
        module=module,
        optimizer=optimizer,
        broadcast_buffers=broadcast_buffers,
        delay_reduce=delay_reduce,
        hierarchical_reduce=hierarchical_reduce,
        message_size=message_size,
        **kwargs,
    ).switch_to(distributed_algorithm)

    return module, optimizer
