from bagua.torch_api.communication import _get_global_state, broadcast
from typing import List


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
        self.buckets = self.algorithm.init_buckets(self.module, self.optimizer)
        self.hooks = self.algorithm.init_hooks(self.module, self.optimizer)
        for bucket in self.buckets:
            self.algorithm.init_operations(
                bucket,
                self.inter_node_communicator,
                self.intra_node_communicator,
                self.global_communicator,
            )
