import torch
from bagua.torch_api.algorithms.gradient_allreduce import GradientAllReduceAlgorithm
from torch.nn.modules import Module
from contextlib import contextmanager
from bagua.torch_api.communication import (
    get_backend,
    get_hyperparameters_service_client,
    broadcast,
    _get_default_group,
    from_torch_group,
    BaguaProcessGroup,
)
from torch._C._distributed_c10d import ProcessGroup as TorchProcessGroup
from bagua.torch_api.model_parallel.moe import is_moe_param
from typing import Callable, Optional
from .inner_distributed import InnerDistributedDataParallel
from torch.nn.parallel import DistributedDataParallel as TorchDistributedDataParallel
from typing import List


class DistributedDataParallel_V1_9_0_Interface(Module):
    r"""
    PyTorch v1.9.0 DistributedDataParallel interface.
    """

    def __init__(self) -> None:
        super(DistributedDataParallel_V1_9_0_Interface, self).__init__()

    @contextmanager
    def no_sync(self):
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def scatter(self, inputs, kwargs, device_ids):
        raise NotImplementedError

    def to_kwargs(self, inputs, kwargs, device_id):
        raise NotImplementedError

    def gather(self, outputs, output_device):
        raise NotImplementedError

    def train(self, mode=True):
        raise NotImplementedError

    @contextmanager
    def join(
        self,
        divide_by_initial_world_size=True,
        enable=True,
        throw_on_early_termination=False,
    ):
        raise NotImplementedError

    def register_comm_hook(self, state: object, hook: Callable):
        raise NotImplementedError

    def will_sync_module_buffers(self):
        raise NotImplementedError


class DistributedDataParallel_V1_9_0(DistributedDataParallel_V1_9_0_Interface):
    r"""
    PyTorch v1.9.0 DistributedDataParallel interface using bagua backend.
    """

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        # The following bagua parameters
        optimizers: List[torch.optim.Optimizer] = [],
        algorithm: "bagua.torch_api.algorithms.Algorithm" = GradientAllReduceAlgorithm(),
    ) -> None:
        super(DistributedDataParallel_V1_9_0, self).__init__()
        assert any((p.requires_grad for p in module.parameters())), (
            "DistributedDataParallel is not needed when a module "
            "doesn't have any parameter that requires a gradient."
        )

        if device_ids is not None and len(device_ids) > 1:
            raise ValueError("device_ids can only be None or contain a single element.")

        self.is_multi_device_module = len({p.device for p in module.parameters()}) > 1
        distinct_device_types = {p.device.type for p in module.parameters()}
        if len(distinct_device_types) != 1:
            raise ValueError(
                "DistributedDataParallel's input module must be on "
                "the same type of devices, but input module parameters locate in {}.".format(
                    distinct_device_types
                )
            )
        self.device_type = list(distinct_device_types)[0]

        self.static_graph = False
        self.dim = dim
        self.module = module
        self.device = list(self.module.parameters())[0].device
        assert broadcast_buffers is True, "Not yet supported"
        self.broadcast_buffers = broadcast_buffers
        assert find_unused_parameters is False, "Not yet supported"
        self.find_unused_parameters = find_unused_parameters

        bagua_process_group = None
        if bagua_process_group is None:
            bagua_process_group = _get_default_group()
        elif type(bagua_process_group) is TorchProcessGroup:
            bagua_process_group = process_group.bagua_patch().bagua_pg
        elif type(bagua_process_group) is BaguaProcessGroup:
            bagua_process_group = process_group

        self.inner_ddp = InnerDistributedDataParallel(
            self.module, optimizers, algorithm, bagua_process_group
        )

    @property
    def require_backward_grad_sync(self):
        return self.inner_ddp.require_backward_grad_sync

    @property
    def parameters_to_ignore(self):
        return self.inner_ddp.parameters_to_ignore

    def forward(self, *inputs, **kwargs):
        output = self.module(*inputs, **kwargs)
        return output

    @contextmanager
    def no_sync(self):
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.inner_ddp.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.inner_ddp.require_backward_grad_sync = old_require_backward_grad_sync

    @property
    def bagua_optimizers(self):
        return self.inner_ddp.bagua_optimizers

    @property
    def inner(self):
        return self.inner_ddp

    def switch_bagua_setting(
        self,
        process_group=None,
        optimizers: List[torch.optim.Optimizer] = [],
        algorithm: "bagua.torch_api.algorithms.Algorithm" = GradientAllReduceAlgorithm(),
    ):
        bagua_process_group = None
        if bagua_process_group is None:
            bagua_process_group = _get_default_group()
        elif type(bagua_process_group) is TorchProcessGroup:
            bagua_process_group = process_group.bagua_patch().bagua_pg
        elif type(bagua_process_group) is BaguaProcessGroup:
            bagua_process_group = process_group

        self.inner_ddp = InnerDistributedDataParallel(
            self.module,
            optimizers=optimizers,
            algorithm=algorithm,
            process_group=bagua_process_group,
            bagua_module_name=self.inner_ddp.bagua_module_name,
        )

    @property
    def bagua_algorithm(self):
        return self.inner_ddp.bagua_algorithm


def DistributedDataParallel(
    module,
    device_ids=None,
    output_device=None,
    dim=0,
    broadcast_buffers=True,
    process_group=None,
    bucket_cap_mb=25,
    find_unused_parameters=False,
    check_reduction=False,
    gradient_as_bucket_view=False,
    # The following bagua parameters
    optimizers: List[torch.optim.Optimizer] = [],
    algorithm: "bagua.torch_api.algorithms.Algorithm" = GradientAllReduceAlgorithm()
):
    fallback_pass = [
        device_ids is not None,
        output_device is not None,
        dim != 0,
        broadcast_buffers is True,
        find_unused_parameters is True,
        gradient_as_bucket_view is True,
        check_reduction is True,
    ]
    if any(fallback_pass):
        return TorchDistributedDataParallel(
            module=module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            process_group=process_group,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            check_reduction=check_reduction,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

    return DistributedDataParallel_V1_9_0(
        module=module,
        device_ids=device_ids,
        output_device=output_device,
        dim=dim,
        broadcast_buffers=broadcast_buffers,
        process_group=process_group,
        bucket_cap_mb=bucket_cap_mb,
        find_unused_parameters=find_unused_parameters,
        check_reduction=check_reduction,
        gradient_as_bucket_view=gradient_as_bucket_view,
        optimizers=optimizers,
        algorithm=algorithm,
    )
