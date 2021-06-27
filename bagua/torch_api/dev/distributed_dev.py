from bagua.torch_api.utils import to_bagua_datatype, average_by_removing_extreme_values
from bagua.torch_api.env import get_autotune_level, get_rank
from bagua.bagua_define import TensorDeclaration
from bagua.torch_api.communication import _get_global_state, broadcast, get_hyperparameters_service_client, get_bagua_hyperparameters
from typing import List, OrderedDict
from types import ModuleType
import gorilla
import time
import logging
import torch
import torch.nn
import copy

@gorilla.patches(torch.nn.Module, filter=lambda name, obj: "bagua" in name)
class DistributedWrapper:
    def _bagua_get_module_params_and_buffers(self):
        if hasattr(self, "_ddp_params_and_buffers_to_ignore"): # TODO: document this
            parameters_to_ignore = self._ddp_params_and_buffers_to_ignore
        else:
            parameters_to_ignore = []
        module_states = []
        for name, param in self.state_dict().items():
            if name not in parameters_to_ignore:
                module_states.append(param)
        return module_states

    def _bagua_broadcast_parameters(self):
        """
        Broadcast model and optimizer states.
        """
        module_states = self._bagua_get_module_params_and_buffers()
        for state in module_states:
            broadcast(state, root=0)
        for optimizer in self.bagua_optimizers:
            optimizer_state_dict = optimizer.state_dict()['state']
            for state in optimizer_state_dict.values():
                for inner_state in state.values():
                    if isinstance(inner_state, torch.Tensor): # TODO: consider the case where this is a scalar
                        broadcast(inner_state, root=0)

    def _bagua_autotune_step(self):
        CYCLE_STEP = 100
        start_time = time.time()

        if self._bagua_train_step_counter != 0 and self._bagua_train_step_counter % CYCLE_STEP == 0:

            # calculate metrics
            self._bagua_autotune_score_record_list.append(
                CYCLE_STEP / float(time.time() - self._bagua_autotune_last_report_time)
            )
            if len(self._bagua_autotune_score_record_list) == 0:
                iter_per_seconds = 0.0
            else:
                iter_per_seconds = sum(self._bagua_autotune_score_record_list) / len(self._bagua_autotune_score_record_list)
            logging.debug("score_record_list={}".format(self._bagua_autotune_score_record_list))
            denoised_iter_per_seconds, std, _ = average_by_removing_extreme_values(
                self._bagua_autotune_score_record_list
               )
            logging.debug("iter_per_seconds={}, denoised_iter_per_seconds={}, std={}"
                          .format(iter_per_seconds, denoised_iter_per_seconds, std))

            # report metrics
            # TODO: @shjwudp add support for reporting tensor completion order so that the autotune service does not
            # rely on tensor registration order
            self._bagua_autotune_client.report_metrics(
                rank=get_rank(),
                unix_timestamp=time.time(),
                train_iter=self._bagua_train_step_counter,
                iter_per_seconds=iter_per_seconds,
                denoised_iter_per_seconds=denoised_iter_per_seconds,
                hyperparameters=get_bagua_hyperparameters().dict(),
            )

            # update parameters
            self._bagua_reset_algorithm()
            self._bagua_autotune_score_record_list.clear()
            self._bagua_autotune_last_report_time = time.time()

        logging.info("autotune overhead={}".format(time.time() - start_time))


    def with_bagua(self, optimizers, algorithm):
        # TODO: do we need to check whether optimizers and model parameters are the same?
        self.bagua_optimizers = optimizers
        self.bagua_algorithm = algorithm
        self._bagua_train_step_counter = 0
        self._bagua_autotune_score_record_list = []
        self._bagua_autotune_last_report_time = time.time()
        self._bagua_autotune_completed = False

        def autotune_hook(self, input):
            if self.training:
                self._bagua_train_step_counter += 1
                if get_autotune_level() >= 1 and not self._bagua_autotune_completed:
                    self._bagua_autotune_step()

        self._bagua_autotune_hook = self.register_forward_pre_hook(
            autotune_hook
        )

        # get communicators
        self._bagua_inter_node_communicator = _get_global_state().get_internode_communicator()
        self._bagua_intra_node_communicator = _get_global_state().get_intranode_communicator()
        self._bagua_global_communicator = _get_global_state().get_global_communicator()

        self._bagua_broadcast_parameters()

        # autotune service
        self._bagua_autotune_client = get_hyperparameters_service_client()

        self._bagua_init_algorithm()
        return self

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
            for tensor_group in self._bagua_tensor_groups for tensor in tensor_group
        ]
        bagua_tensor_group_info = dict(
            [(tensor.bagua_tensor_name, i) for i, tensor_group in enumerate(self._bagua_tensor_groups) for tensor in tensor_group]
        )

        self._bagua_autotune_client.register_models(  # TODO: @shjwudp rename to register tensors
            autotune_tensor_list,
            bagua_tensor_group_info
        ).json()  # TODO: @shjwudp error check


    def _bagua_autotune_get_buckets(self):
        response = self._bagua_autotune_client.ask_hyperparameters(
            rank=get_rank(), train_iter=self._bagua_train_step_counter
        ).json()

        get_bagua_hyperparameters().update(  # TODO: @shjwudp do we need global hyperparameters?
            response["recommended_hyperparameters"]
        )

        self._bagua_autotune_completed = not response["is_autotune_processing"] # TODO: @shjwudp rename this to is autotune completed
        recommended_buckets = map(
            lambda x: list(map(lambda y: self._bagua_tensor_map[y['name']], x)),
            response['recommended_hyperparameters']['buckets'])
        return list(recommended_buckets)


    def _bagua_init_algorithm(self):
        self._bagua_hooks = []
        self._bagua_buckets = []
        self._bagua_tensor_groups = self.bagua_algorithm.init_tensors(self)
        self._bagua_tensor_map = dict(
            [(tensor.bagua_tensor_name, tensor)
             for tensor_group in self._bagua_tensor_groups
             for tensor in tensor_group]
        )
        self._bagua_autotune_register_tensors()
        self._bagua_reset_algorithm()

    def _bagua_reset_algorithm(self):
        # cleanup
        for hook in self._bagua_hooks:
            hook.remove()
        self._bagua_hooks.clear()
        self._bagua_buckets.clear()

        # reset
        raw_buckets = self._bagua_autotune_get_buckets()
        self._bagua_buckets.extend(self.bagua_algorithm.tensors_to_buckets(raw_buckets))
        self._bagua_hooks.extend(self.bagua_algorithm.init_hooks(self))
        for bucket in self._bagua_buckets:
            self.bagua_algorithm.init_operations(
                bucket,
                self._bagua_inter_node_communicator,
                self._bagua_intra_node_communicator,
                self._bagua_global_communicator,
            )
        # TODO: @shaoduogan add bucket registration
