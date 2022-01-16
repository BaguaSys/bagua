# pytype: disable=attribute-error
import torch
import time
import io
import pickle
import collections
import logging
from typing import List, Tuple, Optional, Dict
from torch.nn.modules import Module

import bagua
from bagua.torch_api import env
from bagua.torch_api.communication import (
    get_backend,
    get_hyperparameters_service_client,
    broadcast,
    BaguaProcessGroup,
)
from bagua.torch_api.model_parallel.moe import is_moe_param
from bagua.bagua_define import (
    TensorDeclaration,
    BaguaHyperparameter,
)
from bagua.torch_api.utils import to_bagua_datatype, StatisticalAverage


class BaguaDistributedDataParallel:
    def __init__(
        self,
        module: Module,
        optimizers: List[torch.optim.Optimizer],
        algorithm: "bagua.torch_api.algorithms.Algorithm",
        process_group: BaguaProcessGroup,
        bagua_module_name: Optional[str] = None,
        gradient_as_bucket_view: bool = True,
        find_unused_parameters: bool = False,
    ) -> None:
        self.module = module
        self.bagua_module_name = bagua_module_name

        self.bagua_optimizers = optimizers
        self.bagua_algorithm = algorithm.reify(process_group)
        self.process_group = process_group
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.find_unused_parameters = find_unused_parameters
        self.parameters_to_ignore = (
            []
        )  #: the parameter names to ignore during communication
        if hasattr(self.module, "_bagua_params_and_buffers_to_ignore"):
            self.parameters_to_ignore.extend(
                self.module._bagua_params_and_buffers_to_ignore
            )
        if hasattr(
            self.module, "_ddp_params_and_buffers_to_ignore"
        ):  # for compatibility with PyTorch DDP
            self.parameters_to_ignore.extend(
                self.module._ddp_params_and_buffers_to_ignore
            )

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

        class BaguaDistributedDataParallelStates:
            """Empty class whose instances are used for keeping track of BaguaDistributedDataParallel's internal states."""

            pass

        if hasattr(self.module, "_bagua_states"):
            self._reset_algorithm_state()

        self.module._bagua_states = BaguaDistributedDataParallelStates()
        bagua_states = self.module._bagua_states
        bagua_states._bagua_autograd_hooks = []
        bagua_states._bagua_framework_hooks = []

        self._bagua_backend = get_backend(self.bagua_module_name)
        self._bagua_hyperparameters = BaguaHyperparameter()
        self._speed_metrics_switch_on = env.get_autotune_level() >= 1
        self._speed_metrics = StatisticalAverage()
        self.require_backward_grad_sync = True
        self.autograd_graph_params: Dict[str, torch.nn.Parameter] = {}

        ddp = self

        def autotune_hook(self, input):
            if self.training:
                if env.get_autotune_level() >= 1 and not ddp._bagua_autotune_completed:
                    ddp._bagua_autotune_step()

        def clear_post_backward_callback_queued_hook(self, input):
            ddp._is_post_backward_callback_queued = False

        def num_iteration_step_hook(self, input):
            if self.training:
                ddp.bagua_train_step_counter += 1

        def algorithm_reset_hook(self, input):
            if ddp.bagua_algorithm.need_reset() and self.training:
                ddp._bagua_init_algorithm()

        def algorithm_forward_pre_hook(self, input):
            if self.training:
                ddp.bagua_algorithm.init_forward_pre_hook(ddp)(input)

        def record_speed_metrics_event(self, _):
            if not ddp._speed_metrics_switch_on:
                return

            if hasattr(ddp, "_last_event_pair"):
                (start, stop) = ddp._last_event_pair
                try:
                    elapsed_time_s = start.elapsed_time(stop) / 1000.0
                    total_bytes = sum(bucket.bytes() for bucket in ddp.bagua_buckets)
                    total_gbytes = total_bytes / 1024.0 ** 3
                    speed = total_gbytes / elapsed_time_s
                    ddp._speed_metrics.record(speed)
                except RuntimeError as err:
                    logging.debug("Ignore cuda err={}".format(err))

            start_event = torch.cuda.Event(enable_timing=True)
            ddp._speed_metrics_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.current_stream().record_event(start_event)
            ddp._last_event_pair = (start_event, ddp._speed_metrics_end_event)

        def clear_autograd_graph_params(self, _):
            ddp.autograd_graph_params.clear()

        bagua_states._bagua_framework_hooks.extend(
            [
                self.module.register_forward_pre_hook(clear_autograd_graph_params),
                self.module.register_forward_pre_hook(num_iteration_step_hook),
                self.module.register_forward_pre_hook(algorithm_reset_hook),
                self.module.register_forward_pre_hook(algorithm_forward_pre_hook),
                self.module.register_forward_pre_hook(record_speed_metrics_event),
                self.module.register_forward_pre_hook(autotune_hook),
                self.module.register_forward_pre_hook(
                    clear_post_backward_callback_queued_hook
                ),
            ]
        )

        # autotune service
        self._bagua_autotune_client = get_hyperparameters_service_client()

        self._bagua_init_algorithm()

    def bagua_build_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """
        Build tuple of ``(parameter_name, parameter)`` for all parameters that
        require grads and not in the ``_bagua_params_and_buffers_to_ignore`` attribute.
        """
        modules_and_parameters = [
            (module, parameter)
            for module_name, module in self.module.named_modules()
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

        if self.find_unused_parameters and len(self.autograd_graph_params) != 0:
            modules_and_parameters = filter(
                lambda it: it[1][0] in self.autograd_graph_params,
                modules_and_parameters,
            )

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
                        p.bagua_ensure_grad()
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
            broadcast(param, src=0, comm=self.process_group.get_global_communicator())
        scalars = self._bagua_broadcast_scalars(scalars, src=0)
        for key, p in scalars.items():
            call_back_param[key](p)

    def _bagua_broadcast_scalars(self, scalars, src):
        # Serializes and broadcast scalars by converting them to "ByteTensor".
        b = io.BytesIO()
        pickle.dump(scalars, b)
        t = torch.ByteTensor(bytearray(b.getvalue())).cuda()
        broadcast(t, src=0, comm=self.process_group.get_global_communicator())
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
            broadcast(state, src=0, comm=self.process_group.get_global_communicator())
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
            self._reset_buckets()
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
        self._bagua_broadcast_parameters()

        self._bagua_tensors = self.bagua_algorithm.init_tensors(self)
        self._bagua_tensor_map = dict(
            [(tensor.bagua_tensor_name, tensor) for tensor in self._bagua_tensors]
        )
        self._bagua_autotune_register_tensors()
        self._reset_buckets()

        self._register_autograd_hooks()
        self._register_optimizer_hooks()

    def _delay_allreduce(self):
        for param_name, parameter in self.bagua_build_params():
            self.bagua_algorithm.init_backward_hook(self)(param_name, parameter)
            self.bagua_algorithm.init_post_backward_hook(self)()

    def _cleanup_autograd_hooks(self):
        bagua_states = self.module._bagua_states
        for hook in bagua_states._bagua_autograd_hooks:
            hook.remove()
        bagua_states._bagua_autograd_hooks.clear()

    def _register_autograd_hooks(self):
        bagua_states = self.module._bagua_states
        self._cleanup_autograd_hooks()

        for name, param in self.module.named_parameters():

            def real_hook_factory(param_name, parameter):
                def real_hook(*unused):
                    if not self.require_backward_grad_sync:
                        return

                    if self.find_unused_parameters:
                        self.autograd_graph_params[param_name] = parameter

                    self.bagua_algorithm.init_backward_hook(self)(param_name, parameter)

                    def real_post_backward_hook(*unused):
                        self.bagua_algorithm.init_post_backward_hook(self)()
                        if self._speed_metrics_switch_on:
                            torch.cuda.current_stream().record_event(
                                self._speed_metrics_end_event
                            )

                        if self.find_unused_parameters:
                            if (
                                set(self.autograd_graph_params.keys())
                                != self.params_in_use
                            ):
                                self._reset_buckets()
                                self._delay_allreduce()

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
                bagua_states._bagua_autograd_hooks.append(hook)

    def _register_optimizer_hooks(self):
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

    def _reset_buckets(self):
        raw_buckets = self._bagua_autotune_get_buckets()
        self.bagua_buckets = self.bagua_algorithm.tensors_to_buckets(
            raw_buckets, self.gradient_as_bucket_view
        )
        for bucket in self.bagua_buckets:
            self.bagua_algorithm.init_operations(
                self,
                bucket,
            )
        self._bagua_backend.register_ordered_buckets(
            [bucket.backend_bucket for bucket in self.bagua_buckets]
        )
        self.params_in_use = set([name for name, _ in self.bagua_build_params()])

    def _reset_algorithm_state(self):
        bagua_states = self.module._bagua_states
        if hasattr(bagua_states, "_bagua_framework_hooks"):
            for hook in bagua_states._bagua_framework_hooks:
                hook.remove()

        if hasattr(bagua_states, "_bagua_autograd_hooks"):
            self._cleanup_autograd_hooks()
