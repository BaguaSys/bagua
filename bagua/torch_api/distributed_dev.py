from bagua.torch_api.communication import _get_global_state, broadcast


class Algorithm:
    def __init__(self, ):
        pass

    def register_buckets(self, module, optimizer):
        pass

    def register_hooks(self, module, optimizer):
        pass

    def setup_operations(self, bucket, inter_node_communicator, intra_node_communicator, global_communicator):
        pass


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


def broadcast_parameters(module):
    module_states = _get_module_params_and_buffers(module)
    authoritative_rank = 0
    for state in module_states:
        broadcast(state, root=authoritative_rank)


class DistributedWrapper():
    def __init__(self, module, optimizer, algorithm):
        self.module = module
        self.optimizer = optimizer
        self.algorithm = algorithm

        # get communicators
        self.inter_node_communicator = _get_global_state().get_internode_communicator()
        self.intra_node_communicator = _get_global_state().get_intranode_communicator()
        self.global_communicator = _get_global_state().get_global_communicator()

        broadcast_parameters(self.module)
        # TODO: broadcast optimizer parameters

        self.buckets = self.algorithm.register_buckets(self.module, self.optimizer)
        self.algorithm.register_hooks(self.module, self.optimizer)
        for bucket in self.buckets:
            self.algorithm.setup_operations(bucket, self.inter_node_communicator, self.intra_node_communicator,
                                            self.global_communicator)