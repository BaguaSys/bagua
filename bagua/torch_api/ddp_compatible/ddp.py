import torch
import time
import os
import io
import pickle
import collections
import logging
import warnings
import itertools
import bagua.torch_api as bagua
from bagua.torch_api.algorithms import gradient_allreduce
from torch.nn.modules import Module
from contextlib import contextmanager
from typing import List, Tuple, Optional
from bagua.torch_api import env
from bagua.torch_api.communication import (
    get_backend,
    get_hyperparameters_service_client,
    broadcast,
    _get_default_group,
    BaguaProcessGroup,
)
from bagua.bagua_define import (
    TensorDeclaration,
    BaguaHyperparameter,
)
from bagua.torch_api.utils import to_bagua_datatype, StatisticalAverage


# TODO @shjwudp: make DistributedDataParallel_V1_9_0 interface!

class DistributedDataParallel_V1_9_0(Module):
    r"""
    PyTorch v1.9.0 DistributedDataParallel interface.
    """
    pass


class BaguaDistributedDataParallel_V1_9_0(DistributedDataParallel_V1_9_0):
    r"""
    PyTorch v1.9.0 DistributedDataParallel interface using bagua backend.
    """
    pass


class DistributedDataParallel_V1_9_0(Module):
    r"""
    Pytorch DDP using bagua backend
    """

    __id_iter = itertools.count()

    def bagua_init(
        self,
        optimizers: List[torch.optim.Optimizer],
        algorithm: "bagua.torch_api.algorithms.Algorithm",
        process_group: Optional[BaguaProcessGroup] = None
    ):
        self.bagua_module_name = "{}_{}".format(
            self.__class__.__name__, next(DistributedDataParallel_V1_9_0.__id_iter)
        )

        self.bagua_optimizers = optimizers
        self.bagua_algorithm = algorithm.reify()
        self.parameters_to_ignore = (
            []
        )  #: the parameter names to ignore during communication
        if hasattr(self, "_bagua_params_and_buffers_to_ignore"):
            self.parameters_to_ignore.extend(self._bagua_params_and_buffers_to_ignore)
        if hasattr(
            self, "_ddp_params_and_buffers_to_ignore"
        ):  # for compatibility with PyTorch DDP
            self.parameters_to_ignore.extend(self._ddp_params_and_buffers_to_ignore)

        self.bagua_train_step_counter = 0
        """
        Number of iterations in training mode.
        """
        self.bagua_buckets = []
        """
        All Bagua buckets in a list.
        """
        self._bagua_autotune_last_report_time = time.time()
        self._bagua_autotune_completed = False
        self._bagua_framework_hooks = (
            []
        )  # hooks for bagua framework logic, not cleared when changing algorithms
        self._bagua_algorithm_hooks = []
        self._bagua_backend = get_backend(self.bagua_module_name)
        self._bagua_hyperparameters = BaguaHyperparameter()
        self._speed_metrics_switch_on = env.get_autotune_level() >= 1
        self._speed_metrics = StatisticalAverage()

        def autotune_hook(self, input):
            if self.training:
                if env.get_autotune_level() >= 1 and not self._bagua_autotune_completed:
                    self._bagua_autotune_step()

        def clear_post_backward_callback_queued_hook(self, input):
            self._is_post_backward_callback_queued = False

        def num_iteration_step_hook(self, input):
            if self.training:
                self.bagua_train_step_counter += 1

        def algorithm_reset_hook(self, input):
            if self.bagua_algorithm.need_reset():
                self._bagua_init_algorithm()

        def algorithm_forward_pre_hook(self, input):
            if self.training:
                self.bagua_algorithm.init_forward_pre_hook(self)(input)

        def record_speed_metrics_event(self, _):
            if not self._speed_metrics_switch_on:
                return

            if hasattr(self, "_last_event_pair"):
                (start, stop) = self._last_event_pair
                try:
                    elapsed_time_s = start.elapsed_time(stop) / 1000.0
                    total_bytes = sum(bucket.bytes() for bucket in self.bagua_buckets)
                    total_gbytes = total_bytes / 1024.0 ** 3
                    speed = total_gbytes / elapsed_time_s
                    self._speed_metrics.record(speed)
                except RuntimeError as err:
                    logging.debug("Ignore cuda err={}".format(err))

            start_event = torch.cuda.Event(enable_timing=True)
            self._speed_metrics_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.current_stream().record_event(start_event)
            self._last_event_pair = (start_event, self._speed_metrics_end_event)

        self._bagua_framework_hooks.extend(
            [
                self.register_forward_pre_hook(num_iteration_step_hook),
                self.register_forward_pre_hook(algorithm_reset_hook),
                self.register_forward_pre_hook(algorithm_forward_pre_hook),
                self.register_forward_pre_hook(record_speed_metrics_event),
                self.register_forward_pre_hook(autotune_hook),
                self.register_forward_pre_hook(
                    clear_post_backward_callback_queued_hook
                ),
            ]
        )

        # set bucket process group
        if process_group is None:
            self._bagua_process_group = _get_default_group()
        else:
            self._bagua_process_group = process_group

        # autotune service
        self._bagua_autotune_client = get_hyperparameters_service_client()

        self._bagua_init_algorithm()
        return self

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        # The following bagua parameters
        optimizers: List[torch.optim.Optimizer] = [],
        algorithm: "bagua.torch_api.algorithms.Algorithm" = gradient_allreduce.GradientAllReduceAlgorithm(),
    ) -> None:
        super(DistributedDataParallel_V1_9_0, self).__init__()

        # assert any((p.requires_grad for p in module.parameters())), (
        #     "DistributedDataParallel is not needed when a module "
        #     "doesn't have any parameter that requires a gradient."
        # )

        # if device_ids is not None and len(device_ids) > 1:
        #     raise ValueError("device_ids can only be None or contain a single element.")

        # self.is_multi_device_module = len({p.device for p in module.parameters()}) > 1
        # distinct_device_types = {p.device.type for p in module.parameters()}
        # if len(distinct_device_types) != 1:
        #     raise ValueError(
        #         "DistributedDataParallel's input module must be on "
        #         "the same type of devices, but input module parameters locate in {}.".format(
        #             distinct_device_types
        #         )
        #     )
        # self.device_type = list(distinct_device_types)[0]

        # if process_group is not None:
        #     assert type(process_group) == BaguaProcessGroup

        # self.static_graph = False
        # self.dim = dim
        # self.module = module
        # self.device = list(self.module.parameters())[0].device
        # self.broadcast_buffers = broadcast_buffers
        # self.find_unused_parameters = find_unused_parameters
        # self.require_backward_grad_sync = True
        # self.require_forward_param_sync = True
        # self.gradient_as_bucket_view = gradient_as_bucket_view
        # if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
        #     self.parameters_to_ignore = module._ddp_params_and_buffers_to_ignore
        # else:
        #     self.parameters_to_ignore = []

        # if check_reduction:
        #     # This argument is no longer used since the reducer
        #     # will ensure reduction completes even if some parameters
        #     # do not receive gradients.
        #     warnings.warn(
        #         "The `check_reduction` argument in `DistributedDataParallel` "
        #         "module is deprecated. Please avoid using it."
        #     )

        # # Check that a module does not have Uninitialized parameters
        # for param in module.parameters():
        #     if isinstance(param, torch.nn.parameter.UninitializedParameter):
        #         raise RuntimeError(
        #             "Modules with uninitialized parameters can't be used with `DistributedDataParallel`. "
        #             "Run a dummy forward pass to correctly initialize the modules"
        #         )
        # # used for intra-node param sync and inter-node sync as wel
        # self.broadcast_bucket_size = int(250 * 1024 * 1024)

        # # reduction bucket size
        # self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)
        # # Whether to perform input tensor CPU to GPU copies on a side-stream
        # self.use_side_stream_for_tensor_copies = (
        #     os.environ.get("PYTORCH_DDP_USE_SIDE_STREAM", "1") == "1"
        # )

        # # TODO(wayi@): Remove this field since SPMD is no longer supported,
        # # and also remove all the relevant unnecessary loops.
        # # Module replication within process (single-process multi device)
        # self._module_copies = [self.module]

        self.bagua_init(optimizers, algorithm, process_group)

    @property
    def device_ids():
        raise NotImplementedError

    @property
    def output_device():
        raise NotImplementedError

    @property
    def ddp_uneven_inputs_config():
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        # if self.training:
        #     # num_iteration_step_hook
        #     self.bagua_train_step_counter += 1

        #     # algorithm_reset_hook
        #     if self.bagua_algorithm.need_reset():
        #         self._bagua_init_algorithm()

        #     # algorithm_forward_pre_hook
        #     self.bagua_algorithm.init_forward_pre_hook(self)(input)

        #     # record_speed_metrics_event
        #     if self._speed_metrics_switch_on:
        #         if hasattr(self, "_last_event_pair"):
        #             (start, stop) = self._last_event_pair
        #             try:
        #                 elapsed_time_s = start.elapsed_time(stop) / 1000.0
        #                 total_bytes = sum(bucket.bytes() for bucket in self.bagua_buckets)
        #                 total_gbytes = total_bytes / 1024.0 ** 3
        #                 speed = total_gbytes / elapsed_time_s
        #                 self._speed_metrics.record(speed)
        #             except RuntimeError as err:
        #                 logging.debug("Ignore cuda err={}".format(err))

        #         start_event = torch.cuda.Event(enable_timing=True)
        #         self._speed_metrics_end_event = torch.cuda.Event(enable_timing=True)
        #         torch.cuda.current_stream().record_event(start_event)
        #         self._last_event_pair = (start_event, self._speed_metrics_end_event)

        #     # autotune_hook
        #     if env.get_autotune_level() >= 1 and not self._bagua_autotune_completed:
        #         self._bagua_autotune_step()

        #     # clear_post_backward_callback_queued_hook
        #     self._is_post_backward_callback_queued = False

        output = self.module(*inputs, **kwargs)

        return output

    def bagua_build_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """
        Build tuple of ``(parameter_name, parameter)`` for all parameters that
        require grads and not in the ``_bagua_params_and_buffers_to_ignore`` attribute.
        """
        modules_and_parameters = [
            (module, parameter)
            for module_name, module in self.named_modules()
            for parameter in [
                (f"{module_name}.{param_name}", param)
                # Note that we access module.named_parameters instead of
                # parameters(module). parameters(module) is only needed in the
                # single-process multi device case, where it accesses replicated
                # parameters through _former_parameters.
                for param_name, param in module.named_parameters(recurse=False)
                if param.requires_grad
                and f"{module_name}.{param_name}" not in self.parameters_to_ignore
                and (not is_moe_param(param))
            ]
        ]

        # Deduplicate any parameters that might be shared across child modules.
        memo = set()
        # "p not in memo" is the deduplication check.
        # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
        modules_and_parameters = [
            (m, p)
            for m, p in modules_and_parameters
            if p[1] not in memo and not memo.add(p[1])
        ]

        # Build list of parameters.
        parameters = [parameter for _, parameter in modules_and_parameters]

        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, torch.nn.Embedding) or isinstance(
                module, torch.nn.EmbeddingBag
            ):
                return module.sparse
            return False

        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = [
            produces_sparse_gradient(module) for module, _ in modules_and_parameters
        ]

        if any(expect_sparse_gradient):
            raise NotImplementedError("sparse gradient not supported yet")

        return parameters

    # Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
    # Copyright (c) 2021 Kuaishou AI Platform & DS3 Lab.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================
    def _bagua_broadcast_optimizer_state(self, optimizer):
        # L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values.
        if isinstance(optimizer, torch.optim.LBFGS):
            raise ValueError("cannot broadcast torch.optim.LBFGS state")
        optimizer_state_dict = optimizer.state_dict()

        # Initialize newly created optimizers.
        if len(optimizer_state_dict["state"]) == 0:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.requires_grad and id(p) not in optimizer_state_dict["state"]:
                        p.grad = p.data.new(p.size()).zero_()
                        if isinstance(optimizer, torch.optim.SparseAdam):
                            p.grad = p.grad.to_sparse()
            optimizer_state_dict = optimizer.state_dict()
        if len(optimizer_state_dict["state"]) == 0:
            return

        def _state_param_callback(param_id, param_name):
            def _assign_state(v):
                optimizer_state_dict["state"][param_id][param_name] = v

            return _assign_state

        def _hyper_param_callback(index, group_key):
            def _assign_hyper(v):
                optimizer.param_groups[index][group_key] = v

            return _assign_hyper

        params = []
        scalars = collections.OrderedDict()
        call_back_param = {}
        repeat_param_count = collections.defaultdict(int)

        # All "sorted()" operations in this module are used to
        # guarteen the scalar's record order samely in differet ranks.
        for index, param_group in enumerate(optimizer_state_dict["param_groups"]):
            for group_key, group_value in sorted(
                param_group.items(), key=lambda item: item[0]
            ):
                # Hyper-parameters like learning rate are scalars, we need to broadcast them separately.
                if group_key != "params":
                    key = "%s_%d" % (group_key, index)
                    scalars[key] = group_value
                    call_back_param[key] = _hyper_param_callback(index, group_key)
            for param_id in sorted(param_group["params"]):
                if param_id not in optimizer_state_dict["state"]:
                    continue
                param_state = optimizer_state_dict["state"][param_id]
                for param_name, inner_state in sorted(
                    param_state.items(), key=lambda item: item[0]
                ):
                    # Some parameter names, e.g., step, may appear more than once, in which
                    # case we ensure they have a unique identifier defined by
                    # their order.
                    repeat_param_count[param_name] += 1
                    key = "%s_%d" % (str(param_name), repeat_param_count[param_name])
                    if isinstance(inner_state, torch.Tensor):
                        params.append((key, inner_state))
                    else:
                        scalars[key] = inner_state
                        call_back_param[key] = _state_param_callback(
                            param_id, param_name
                        )
        for key, param in params:
            broadcast(param, src=0)
        scalars = self._bagua_broadcast_scalars(scalars, src=0)
        for key, p in scalars.items():
            call_back_param[key](p)

    def _bagua_broadcast_scalars(self, scalars, src):
        # Serializes and broadcast scalars by converting them to "ByteTensor".
        b = io.BytesIO()
        pickle.dump(scalars, b)
        t = torch.ByteTensor(bytearray(b.getvalue())).cuda()
        broadcast(t, src=0)
        if env.get_rank() != src:
            buf = io.BytesIO(t.cpu().numpy().tobytes())
            scalars = pickle.load(buf)

        return scalars

    def _bagua_broadcast_parameters(self):
        """
        Broadcast model and optimizer states.
        """

        module_states = self.bagua_build_params()
        for name, state in module_states:
            broadcast(state, src=0)
        for optimizer in self.bagua_optimizers:
            self._bagua_broadcast_optimizer_state(optimizer)

    def _bagua_autotune_step(self):
        CYCLE_STEP = 100
        start_time = time.time()

        if (
            self.bagua_train_step_counter != 0
            and self.bagua_train_step_counter % CYCLE_STEP == 0
        ):
            # get speed metrics
            time_since_last_update = time.time() - self._bagua_autotune_last_report_time
            speed = self._speed_metrics.get(time_since_last_update)

            # report metrics
            rsp = self._bagua_autotune_client.report_metrics(
                model_name=self.bagua_module_name,
                rank=env.get_rank(),
                train_iter=self.bagua_train_step_counter,
                hyperparameters=self._bagua_hyperparameters.dict(),
                speed=speed,
            )
            assert rsp.status_code == 200, "Unexpected rsp={}".format(rsp)

            # update parameters
            self._bagua_reset_algorithm_buckets()
            self._bagua_autotune_last_report_time = time.time()

        logging.debug("autotune overhead=%s", time.time() - start_time)

    def _bagua_autotune_register_tensors(self):
        """
        Register tensors on autotune server, and return first bucketing suggestions
        """
        autotune_tensor_list = [
            TensorDeclaration(
                {
                    "name": tensor.bagua_tensor_name,
                    "num_elements": tensor.numel(),
                    "dtype": to_bagua_datatype(tensor.dtype),
                }
            )
            for tensor in self._bagua_tensors
        ]

        rsp = self._bagua_autotune_client.register_tensors(
            model_name=self.bagua_module_name,
            tensor_list=autotune_tensor_list,
        )
        assert rsp.status_code == 200, "Unexpected rsp={}".format(rsp)

    def _bagua_autotune_get_buckets(self):
        rsp = self._bagua_autotune_client.ask_hyperparameters(
            model_name=self.bagua_module_name,
            rank=env.get_rank(),
            train_iter=self.bagua_train_step_counter,
        )
        assert rsp.status_code == 200, "Unexpected rsp={}".format(rsp)
        recommended_hyperparameters = rsp.json()["recommended_hyperparameters"]
        is_autotune_completed = rsp.json()["is_autotune_completed"]

        self._bagua_hyperparameters.update(recommended_hyperparameters)

        self._bagua_autotune_completed = is_autotune_completed
        recommended_buckets = map(
            lambda x: list(map(lambda y: self._bagua_tensor_map[y["name"]], x)),
            recommended_hyperparameters["buckets"],
        )
        return list(recommended_buckets)

    def _bagua_init_algorithm(self):
        self._bagua_cleanup_algorithm()
        self._bagua_broadcast_parameters()
        self._bagua_tensors = self.bagua_algorithm.init_tensors(self)
        self._bagua_tensor_map = dict(
            [(tensor.bagua_tensor_name, tensor) for tensor in self._bagua_tensors]
        )
        self._bagua_autotune_register_tensors()
        self._bagua_reset_algorithm_buckets()

    def _bagua_cleanup_algorithm(self):
        for hook in self._bagua_algorithm_hooks:
            hook.remove()
        self._bagua_algorithm_hooks.clear()
        self.bagua_buckets.clear()

    def _bagua_reset_algorithm_buckets(self):
        self._bagua_cleanup_algorithm()
        raw_buckets = self._bagua_autotune_get_buckets()
        self.bagua_buckets.extend(self.bagua_algorithm.tensors_to_buckets(raw_buckets))

        for name, param in self.named_parameters():

            def real_hook_factory(param_name, parameter):
                def real_hook(*unused):
                    if not self.require_backward_grad_sync:
                        return

                    self.bagua_algorithm.init_backward_hook(self)(param_name, parameter)

                    def real_post_backward_hook(*unused):
                        self.bagua_algorithm.init_post_backward_hook(self)()
                        if self._speed_metrics_switch_on:
                            torch.cuda.current_stream().record_event(
                                self._speed_metrics_end_event
                            )

                    if not self._is_post_backward_callback_queued:
                        torch.autograd.Variable._execution_engine.queue_callback(
                            real_post_backward_hook
                        )
                        self._is_post_backward_callback_queued = True

                return real_hook

            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                hook = grad_acc.register_hook(real_hook_factory(name, param))
                hook.grad_acc = grad_acc
                self._bagua_algorithm_hooks.append(hook)

        optimizer_hook = self.bagua_algorithm.init_post_optimizer_step_hook(self)

        from types import MethodType

        for optimizer in self.bagua_optimizers:
            if not hasattr(optimizer, "_bagua_original_step"):
                optimizer._bagua_original_step = optimizer.step

            def new_step_factory(optimizer):
                def new_step(self, *args, **kwargs):
                    result = self._bagua_original_step(*args, **kwargs)

                    optimizer_hook(self)
                    return result

                return MethodType(new_step, optimizer)

            optimizer.step = new_step_factory(optimizer)

        for bucket in self.bagua_buckets:
            self.bagua_algorithm.init_operations(
                self,
                bucket,
            )
        self._bagua_backend.register_ordered_buckets(
            [bucket.backend_bucket for bucket in self.bagua_buckets]
        )

    @contextmanager
    def no_sync(self):
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError

    def scatter(self, inputs, kwargs, device_ids):
        raise NotImplementedError

    def to_kwargs(self, inputs, kwargs, device_id):
        raise NotImplementedError

    def gather(self, outputs, output_device):
        raise NotImplementedError

    def train(self, mode=True):
        super(DistributedDataParallel_V1_9_0, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)
        return self

    @contextmanager
    def join(
        self,
        divide_by_initial_world_size=True,
        enable=True,
        throw_on_early_termination=False,
    ):
        raise NotImplementedError

    def register_comm_hook(self, state: object, hook: callable):
        raise NotImplementedError

    def will_sync_module_buffers(self):
        raise NotImplementedError