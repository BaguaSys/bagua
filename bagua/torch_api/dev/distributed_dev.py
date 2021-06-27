from bagua.torch_api.communication import _get_global_state, broadcast
from typing import List, OrderedDict


class DistributedWrapper:
    def _get_module_params_and_buffers(module):
        # TODO: document this
        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            parameters_to_ignore = module._ddp_params_and_buffers_to_ignore
        else:
            parameters_to_ignore = []
        module_states = []
        for name, param in module.state_dict().items():
            if name not in parameters_to_ignore:
                module_states.append(param)
        return module_states

    def _broadcast_parameters(module):
        module_states = DistributedWrapper._get_module_params_and_buffers(module)
        for state in module_states:
            broadcast(state, root=0)

    def __init__(self, module, optimizer, algorithm):
        self.module = module
        self.optimizer = optimizer
        self.algorithm = algorithm

        # get communicators
        self.inter_node_communicator = _get_global_state().get_internode_communicator()
        self.intra_node_communicator = _get_global_state().get_intranode_communicator()
        self.global_communicator = _get_global_state().get_global_communicator()

        DistributedWrapper._broadcast_parameters(self.module)
        # TODO: broadcast optimizer parameters

        self.init_algorithm()

    def init_algorithm(self):
        self.tensors = self.algorithm.init_tensors(self.module, self.optimizer)
        # FIXME
        raw_buckets = []
        for tensor_name, tensor in self.tensors.items():
            raw_buckets.append(OrderedDict([(tensor_name, tensor)]))
        self.buckets = self.algorithm.tensors_to_buckets(raw_buckets)
        self.hooks = self.algorithm.init_hooks(self.module, self.optimizer)
        for bucket in self.buckets:
            self.algorithm.init_operations(
                bucket,
                self.inter_node_communicator,
                self.intra_node_communicator,
                self.global_communicator,
            )
