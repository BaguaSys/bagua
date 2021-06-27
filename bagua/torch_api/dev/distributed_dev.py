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

    def _bagua_broadcast_parameters(self):
        module_states = self._bagua_get_module_params_and_buffers()
        for state in module_states:
            broadcast(state, root=0)
        for optimizer in self.bagua_optimizers:
            optimizer_state_dict = optimizer.state_dict()['state']
            for state in optimizer_state_dict.values():
                for inner_state in state.values():
                    if isinstance(inner_state, torch.Tensor): # TODO: consider the case where this is a scalar
                        broadcast(inner_state, root=0)

    def with_bagua(self, optimizers, algorithm):
        # TODO: do we need to check whether optimizers and model parameters are the same?
        self.bagua_optimizers = optimizers
        self.bagua_algorithm = algorithm

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
        tensor_map = dict([(tensor.bagua_tensor_name, tensor)
                           for tensor_group in self._bagua_tensor_groups for tensor in tensor_group])
        bagua_tensor_group_info = dict(
            [(tensor.bagua_tensor_name, i) for i, tensor_group in enumerate(self._bagua_tensor_groups) for tensor in tensor_group]
        )

        recommended_buckets = map(
            lambda x: list(map(lambda y: tensor_map[y['name']], x)),
            self._bagua_autotune_client.register_models(
                autotune_tensor_list,
                bagua_tensor_group_info
            ).json()['recommended_hyperparameters']['buckets'])
        return list(recommended_buckets)


    def _bagua_init_algorithm(self):
        self._bagua_tensor_groups = self.bagua_algorithm.init_tensors(self)
        raw_buckets = self._bagua_autotune_register_tensors()
        self._bagua_buckets = self.bagua_algorithm.tensors_to_buckets(raw_buckets)
        self._bagua_hooks = self.bagua_algorithm.init_hooks(self)
        for bucket in self._bagua_buckets:
            self.bagua_algorithm.init_operations(
                bucket,
                self._bagua_inter_node_communicator,
                self._bagua_intra_node_communicator,
                self._bagua_global_communicator,
            )
