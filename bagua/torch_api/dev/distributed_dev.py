from bagua.torch_api.utils import to_bagua_datatype
from bagua.bagua_define import TensorDeclaration
from bagua.torch_api.communication import _get_global_state, broadcast, get_hyperparameters_service_client
from typing import List, OrderedDict
from types import ModuleType
import gorilla
import torch
import torch.nn

@gorilla.patches(torch.nn.Module, filter=lambda name, obj: "bagua" in name )
class DistributedWrapper:
    def _bagua_get_module_params_and_buffers(self):
        # TODO: document this
        if hasattr(self, "_ddp_params_and_buffers_to_ignore"):
            parameters_to_ignore = self._ddp_params_and_buffers_to_ignore
        else:
            parameters_to_ignore = []
        module_states = []
        for name, param in self.state_dict().items():
            if name not in parameters_to_ignore:
                module_states.append(param)
        return module_states

    def _bagua_get_parameter_group_info(self):
        """
        Given a optimizer, return a dict containing Param => param_group_id
        """
        param_group_info = {}
        param_groups = [
            group for optimizer in self._bagua_optimizers for group in optimizer.param_groups
           ]
        for i, group in enumerate(param_groups):
            for param in group["params"]:
                param_group_info[param.bagua_tensor_name] = i
        return param_group_info

    def _bagua_broadcast_parameters(self):
        module_states = self._bagua_get_module_params_and_buffers()
        for state in module_states:
            broadcast(state, root=0)

    def with_bagua(self, optimizers, algorithm):
        # TODO: do we need to check whether optimizers and model parameters are the same?
        self._bagua_optimizers = optimizers
        self._bagua_algorithm = algorithm

        # get communicators
        self._bagua_inter_node_communicator = _get_global_state().get_internode_communicator()
        self._bagua_intra_node_communicator = _get_global_state().get_intranode_communicator()
        self._bagua_global_communicator = _get_global_state().get_global_communicator()

        self._bagua_broadcast_parameters()
        # TODO: broadcast optimizer parameters

        # autotune service
        self._bagua_autotune_client = get_hyperparameters_service_client()

        self._bagua_init_algorithm()
        return self

    def _bagua_autotune_register_tensors(self):
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
        rsp = self._bagua_autotune_client.register_models(
            autotune_tensor_list, self._bagua_get_parameter_group_info()
        )
        print(rsp.text)

    def _bagua_init_algorithm(self):
        self._bagua_tensors = self._bagua_algorithm.init_tensors(self)
        # FIXME
        self._bagua_autotune_register_tensors()

        raw_buckets = []
        for tensor in self._bagua_tensors:
            raw_buckets.append([tensor])
        self._bagua_buckets = self._bagua_algorithm.tensors_to_buckets(raw_buckets)
        self._bagua_hooks = self._bagua_algorithm.init_hooks(self)
        for bucket in self._bagua_buckets:
            self._bagua_algorithm.init_operations(
                bucket,
                self._bagua_inter_node_communicator,
                self._bagua_intra_node_communicator,
                self._bagua_global_communicator,
            )
