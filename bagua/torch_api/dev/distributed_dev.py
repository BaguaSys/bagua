from bagua.torch_api.communication import _get_global_state, broadcast
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

    def with_bagua(self, optimizer, algorithm):
        self._bagua_optimizer = optimizer
        self._bagua_algorithm = algorithm

        # get communicators
        self._bagua_inter_node_communicator = _get_global_state().get_internode_communicator()
        self._bagua_intra_node_communicator = _get_global_state().get_intranode_communicator()
        self._bagua_global_communicator = _get_global_state().get_global_communicator()

        self._bagua_broadcast_parameters()
        # TODO: broadcast optimizer parameters

        self._bagua_init_algorithm()

    def _bagua_init_algorithm(self):
        self.tensors = self._bagua_algorithm.init_tensors(self, self._bagua_optimizer)
        # FIXME
        raw_buckets = []
        for tensor_name, tensor in self.tensors.items():
            raw_buckets.append(OrderedDict([(tensor_name, tensor)]))
        self.buckets = self._bagua_algorithm.tensors_to_buckets(raw_buckets)
        self.hooks = self._bagua_algorithm.init_hooks(self, self._bagua_optimizer)
        for bucket in self.buckets:
            self._bagua_algorithm.init_operations(
                bucket,
                self._bagua_inter_node_communicator,
                self._bagua_intra_node_communicator,
                self._bagua_global_communicator,
            )
