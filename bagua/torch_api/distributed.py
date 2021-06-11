import copy
import torch
from torch.autograd import Variable
import logging
from typing import Optional, List, Dict, Tuple
from .. import autotune
from .distributed_define import BucketType
from bagua.torch_api.utils import flatten_module_params
from functools import reduce
from bagua.torch_api.communication import (
    _get_global_state,
    get_bagua_hyperparameters,
    get_hyperparameters_service_client,
)
from bagua.torch_api.env import (
    get_world_size,
    get_rank,
    get_local_size,
    get_autotune_server_addr,
)
from bagua.service.autotune_service import BaguaHyperparameter
from bagua.bagua_define import TensorDtype, TensorDeclaration
import traceback
from .distributed_define import BucketType
from bagua.torch_api.env import get_world_size, get_rank, get_local_size
from .utils import to_bagua_datatype, flatten_module_params, check_contiguous
from bagua.torch_api.communication import _get_global_state
import bagua_core as B


class Reducer(object):
    """
    -- New --
    bucket_initialized is False:
        1. add_param
        2. initialize_buckets -> register_models
        3. mark_bucket_ready
        4. mark_on_complete

    bucket_initialized is True:
        1. mark_tensor_ready
        2. mark_on_complete
    """

    def __init__(
        self,
        module: DistributedModule,
        optimizers: List[torch.optim.Optimizer],
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
        if self.bucket_type == BucketType.Gradient:
            return param.grad.data
        elif self.bucket_type == BucketType.Weight:
            return param.data
        elif self.bucket_type == BucketType.Param:
            return param

    def initialize_buckets(self) -> List[List[torch.Tensor]]:
        """
        NOTE: initialize_buckets MUST execute after the first round of backward for grad ready.
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

        logging.info("Initialized bucktes={}".format(self.buckets))
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
        for bucket_params in self.param_buckets:
            bagua_tensors = [new_bagua_tensor(p) for p in bucket_params]
            # Note: align_bytes does not work when inplace=True
            bagua_bucket = B.BaguaBucketPy(
                bagua_tensors, inplace=self.fusion, align_bytes=self.align_bytes
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
        for param in bucket:
            self.mark_tensor_ready(param)

    def mark_tensor_ready(self, param):
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
        # wait until all buckets are reduced
        self.bagua_backend.wait_pending_comm_ops()

        if self.post_backward_fn is not None:
            self.post_backward_fn(self.bagua_backend)

        self.step_counter += 1


class OverlappingWrapper(torch.nn.Module):
    """
    This class defines the process of communication-computation overlap.

    Parameters:
        * `module`(_DistributedModule_) - A distributed module to enable overlapping.
        * `delay_reduce`(_bool_) - Delay all communication to the end of the
           backward pass. This disables overlapping communication with computation.
        * `bucket_type`(_BucketType_) - Type of elements in a communication bucket, could be
           either module parameters, weights or gradients.
        * `message_size`(_int_) - Minimum bytes in a communication bucket.
        * `hierarchical_reduce`(_bool_) - Enable hierarchical reduce, which will perform an intra-node
           allreduce, followed by an inter-node reduce defined by different `module`, and an intra-node
           broadcast at the end.
        * `decentralize_reduce`(_bool_) - For decentralize training, set `decentralize_reduce` to `True`.
        * `align_bytes`(_int_) - Number to bytes to be aligned for each communication bucket.
        * `chunking`(_bool_) - For scatter-gather communication pattern, set `chunking` to `True`.
        * `fusion`(_bool_) - To reset parameter data pointer so that they can use faster code paths, set
          `fusion` to `True`.

    ..note::
        This implementation benefits a lot from `apex.parallel.DistributedDataParallel`.


    """

    def __init__(
        self,
        module: DistributedModule,
        optimizers: List[torch.optim.Optimizer],
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
        """
        Defines a number of hooks used to reduce communication buckets in backward process.
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
        """
        Overwrites the forward process for a distributed module with communication-computation overlap.
        """
        result = self.module(*inputs, **kwargs)
        self.callback_queued = False
        self.decen_callonce_flag = False
        return result
