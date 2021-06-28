from bagua.torch_api.dev.algorithms import Algorithm
from bagua.torch_api.utils import to_bagua_datatype, average_by_removing_extreme_values
from bagua.torch_api.env import get_autotune_level, get_rank
from bagua.bagua_define import TensorDeclaration
from bagua.torch_api.communication import _get_global_state, broadcast, get_hyperparameters_service_client, get_bagua_hyperparameters
import gorilla
import time
import logging
import torch
import torch.nn

@gorilla.patches(torch.nn.Module, filter=lambda name, obj: "bagua" in name)
class BaguaModule:
    """
    This class patches `torch.nn.Module` with several methods to enable Bagua
    functionalities.
    """

    def _bagua_build_params(self):
        # This function is modified from torch DDP implementation
        # Build tuple of (module, parameter) for all parameters that require grads.
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
                   ]
           ]

        # Deduplicate any parameters that might be shared across child modules.
        memo = set()
        # "p not in memo" is the deduplication check.
        # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
        modules_and_parameters = [(m, p) for m, p in modules_and_parameters if p not in memo and not memo.add(p)]

        # Build list of parameters.
        parameters = [
            parameter for _, parameter in modules_and_parameters
           ]

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

    def _bagua_broadcast_parameters(self):
        """
        Broadcast model and optimizer states.
        """
        module_states = self._bagua_build_params()
        for name, state in module_states:
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
            logging.debug("iter_per_seconds=%s, denoised_iter_per_seconds=%s, std=%s",
                          iter_per_seconds, denoised_iter_per_seconds, std)

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

        logging.info("autotune overhead=%s", time.time() - start_time)


    def with_bagua(self, optimizers: List[torch.optim.Optimizer], algorithm: Algorithm):
        r"""`with_bagua` enables easy distributed data parallel training on a
        `torch.nn.Module`.

        Arguments:
            optimizers: Optimizer(s) used by the
                module. It can contain one or more PyTorch optimizers.
            algorithm: Distributed algorithm used to do the actual communication
                and update.

        Returns:
            The original module, with Bagua related environments initialized.

        .. note::
            If we want to ignore some layers for communication, we can first check
            these layer's corresponding keys in the module's ``state_dict`` (they are
            in ``"{module_name}.{param_name}"`` format), then assign the list of
            keys to ``your_module._bagua_params_and_buffers_to_ignore``.

        Examples::

            >>> model = torch.nn.Sequential(
            ...      torch.nn.Linear(D_in, H),
            ...      torch.nn.ReLU(),
            ...      torch.nn.Linear(H, D_out),
            ...    )
            >>> optimizer = torch.optim.SGD(
            ...      model.parameters(),
            ...      lr=0.01,
            ...      momentum=0.9
            ...    )
            >>> model = model.with_bagua(
            ...      [optimizer],
            ...      GradientAllReduce()
            ...    )
        """

        # TODO: do we need to check whether optimizers and model parameters are the same?
        self.bagua_optimizers = optimizers
        self.bagua_algorithm = algorithm
        self.parameters_to_ignore = []
        if hasattr(self, "_bagua_params_and_buffers_to_ignore"):
            self.parameters_to_ignore.extend(self._bagua_params_and_buffers_to_ignore)
        if hasattr(self, "_ddp_params_and_buffers_to_ignore"): # for compatibility with PyTorch DDP
            self.parameters_to_ignore.extend(self._ddp_params_and_buffers_to_ignore)
        self._bagua_train_step_counter = 0
        self._bagua_autotune_score_record_list = []
        self._bagua_autotune_last_report_time = time.time()
        self._bagua_autotune_completed = False
        self._bagua_framework_hooks = [] # hooks for bagua framework logic, not cleared when changing algorithms
        self._bagua_backend = _get_global_state().get_backend()

        def autotune_hook(self, input):
            if self.training:
                if get_autotune_level() >= 1 and not self._bagua_autotune_completed:
                    self._bagua_autotune_step()

        def clear_post_backward_callback_queued_hook(self, input):
            self._is_post_backward_callback_queued = False

        def num_iteration_step_hook(self, input):
            if self.training:
                self._bagua_train_step_counter += 1

        def safety_check_hook(self, input):
            SAFETY_CHECK_INTERVAL = 1000
            if self._bagua_train_step_counter % SAFETY_CHECK_INTERVAL == 0:
                for bucket in self._bagua_buckets:
                    assert bucket.check_consistence(), \
                    f"""{bucket} found memory inconsistent during training, this could
                    be caused by your coding modifying some module's memory layout
                    after Bagua initialized. Please move such operations before
                    Bagua initialization.
                    """

        self._bagua_framework_hooks.extend(
            [
                self.register_forward_pre_hook(autotune_hook),
                self.register_forward_pre_hook(clear_post_backward_callback_queued_hook),
                self.register_forward_pre_hook(num_iteration_step_hook),
                self.register_forward_pre_hook(safety_check_hook),
             ])

        self.param_i = {}
        parameters = self._bagua_build_params()
        for name, param in parameters:
            self.param_i[id(param)] = name

        self.tensor_events = [
            torch.cuda.Event(enable_timing=False, blocking=False)
        ] * len(self.param_i)

        self.current_stream = torch.cuda.current_stream()
        self.bagua_backend = _get_global_state().get_backend()

        # get communicators
        self.bagua_inter_node_communicator = _get_global_state().get_internode_communicator()
        self.bagua_intra_node_communicator = _get_global_state().get_intranode_communicator()
        self.bagua_global_communicator = _get_global_state().get_global_communicator()

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
        self._bagua_algorithm_hooks = []
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
        for hook in self._bagua_algorithm_hooks:
            hook.remove()
        self._bagua_algorithm_hooks.clear()
        self._bagua_buckets.clear()

        # reset
        raw_buckets = self._bagua_autotune_get_buckets()
        self._bagua_buckets.extend(self.bagua_algorithm.tensors_to_buckets(raw_buckets))

        for name, param in self.named_parameters():
            def real_hook_factory(name):
                def real_hook(*unused):
                    self.bagua_algorithm.init_backward_hook(self)(name)

                    def real_post_backward_hook(*unused):
                        self.bagua_algorithm.init_post_backward_hook(self)()

                    if not self._is_post_backward_callback_queued:
                        torch.autograd.Variable._execution_engine.queue_callback(
                            real_post_backward_hook
                        )
                        self._is_post_backward_callback_queued = True
                return real_hook

            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                hook = grad_acc.register_hook(
                    real_hook_factory(name)
                )
                hook.grad_acc = grad_acc
                self._bagua_algorithm_hooks.append(hook)

        optimizer_hook = self.bagua_algorithm.init_post_step_hook(self)

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

        for bucket in self._bagua_buckets:
            self.bagua_algorithm.init_operations(
                self,
                bucket,
            )
        self._bagua_backend.register_ordered_buckets([bucket.backend_bucket for bucket in self._bagua_buckets])
