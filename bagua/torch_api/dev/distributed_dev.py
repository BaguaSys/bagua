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

    # # TODO: remove parameter group from autotune service

    # def _bagua_get_parameter_group_info(self):
    #     """
    #     Given a optimizer, return a dict containing Param => param_group_id
    #     """
    #     param_group_info = {}
    #     param_groups = [
    #         group for optimizer in self.bagua_optimizers for group in optimizer.param_groups
    #        ]
    #     for i, group in enumerate(param_groups):
    #         for param in group["params"]:
    #             param_group_info[param.bagua_tensor_name] = i
    #     return param_group_info

    def _bagua_broadcast_parameters(self):
        module_states = self._bagua_get_module_params_and_buffers()
        for state in module_states:
            broadcast(state, root=0)

    def with_bagua(self, optimizers, algorithm):
        # TODO: do we need to check whether optimizers and model parameters are the same?
        self.step_counter = 0
        self.bagua_optimizers = optimizers
        self.bagua_algorithm = algorithm

        self.bucket_initialized = False
        self.param_list = []
        self.param_i = {}
        index = 0
        for name, param in self.named_parameters():
            if param.requires_grad and name not in self.parameters_to_ignore:
                self.param_list.append(param)
                self.param_i[name] = index
                index += 1
            else:
                logging.debug(f"skip param: {name}")
        self.tensor_events = [
            torch.cuda.Event(enable_timing=False, blocking=False)
        ] * len(self.param_list)

        self.current_stream = torch.cuda.current_stream()
        self.bagua_backend = _get_global_state().get_backend()

        # get communicators
        self._bagua_inter_node_communicator = _get_global_state().get_internode_communicator()
        self._bagua_intra_node_communicator = _get_global_state().get_intranode_communicator()
        self._bagua_global_communicator = _get_global_state().get_global_communicator()

        self._bagua_broadcast_parameters()
        # TODO: broadcast optimizer parameters

        # autotune service
        self._bagua_autotune_client = get_hyperparameters_service_client()

        self._bagua_init_algorithm()
        self.create_hooks()
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
        self.tensor_map = dict([(tensor.bagua_tensor_name, tensor)
                           for tensor_group in self._bagua_tensor_groups for tensor in tensor_group])
        bagua_tensor_group_info = dict(
            [(tensor.bagua_tensor_name, i) for i, tensor_group in enumerate(self._bagua_tensor_groups) for tensor in tensor_group]
        )

        recommended_buckets = map(
            lambda x: list(map(lambda y: self.tensor_map[y['name']], x)),
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

    def create_hooks(self):
        r"""
        Defines a number of hooks used to reduce communication buckets
        in backward process.
        """
        self.grad_accs = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]

                def make_hook(param, name):

                    def reduce_fallback(skip_reduce=False):
                        if skip_reduce:
                            logging.debug("skip reduce")
                            return
                        for i, bucket in enumerate(self._bagua_buckets):
                            for param in bucket:
                                self.mark_tensor_ready(param.bagua_tensor_name)

                        self.reducer.mark_on_complete()

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

                    def _hook(*unused):
                        if (
                            self.compute_communication_overlap
                        ):  # overlap reduce and backward
                            self.mark_tensor_ready(name)
                            register_post_backward_func(self.mark_on_complete)
                        else:
                            register_post_backward_func(reduce_fallback)

                    return _hook

                h = grad_acc.register_hook(make_hook(param, name))

                self.hook_list.append(h)
                self.grad_accs.append(grad_acc)

    def forward(self, *inputs, **kwargs):
        r"""
        Overwrite the forward process for a distributed module with
        communication-computation overlap.
        """
        result = self.module(*inputs, **kwargs)
        self.callback_queued = False
        return result


    def mark_tensor_ready(self, name):
        r"""
        Mark the tensor ready when got its gradient.
        """
        # param_name = self.param_name[id(param)]
        # if param_name not in self.bagua_tensor:  # no bagua_tensor no mark ready
        #     return

        # if not self.fusion:
        #     # reset bagua tensor pointer
        #     p = self.fill_slot(param)
        #     self.bagua_tensor[param_name].reset_ptr(p.data_ptr())

        ready_event = self.tensor_events[self.param_i[name]]
        self.current_stream.record_event(ready_event)
        self.bagua_backend.mark_communication_ready(
            self.tensor_map[name].backend_tensor, ready_event.cuda_event
        )

    def mark_on_complete(self):
        r"""
        Mark all buckets have finished thier reduce process.
        """
        self.bagua_backend.wait_pending_comm_ops()
        self.step_counter += 1