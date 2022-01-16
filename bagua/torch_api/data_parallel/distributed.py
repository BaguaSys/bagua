# pytype: disable=module-attr
import torch
from torch.nn.modules import Module
from torch.distributed import ProcessGroup as TorchProcessGroup
from torch.nn.parallel import DistributedDataParallel as TorchDistributedDataParallel
from typing import Callable, List, Optional, Union
import warnings

import bagua
from bagua.torch_api.algorithms.gradient_allreduce import GradientAllReduceAlgorithm
from contextlib import contextmanager
from bagua.torch_api.communication import (
    _get_default_group,
    BaguaProcessGroup,
)
from .bagua_distributed import BaguaDistributedDataParallel


class DistributedDataParallel_V1_9_0_Interface(Module):
    r"""
    `PyTorch v1.9.0 DDP <https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/parallel/distributed.py#L125>`_ compatible interface.
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
        super(DistributedDataParallel_V1_9_0_Interface, self).train(mode)
        return self

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


def to_bagua_process_group(
    process_group: Union[TorchProcessGroup, BaguaProcessGroup, None] = None
):
    """Convert a PyTorch process group to a Bagua process group.

    Args:
        process_group (Union[TorchProcessGroup, BaguaProcessGroup, None], optional): PyTorch
            process group or Bagua process group. The default PyTorch process group is used if ``None`` is passed in.

    Raises:
        Exception: raise unexpect input exception if input is not
            ``TorchProcessGroup``, ``BaguaProcessGroup`` or ``None``.

    Returns:
        BaguaProcessGroup: process group for communication in bagua.
    """

    if process_group is None:
        return _get_default_group()
    elif type(process_group) in [
        TorchProcessGroup,
        torch.distributed.ProcessGroupNCCL,
    ]:
        return process_group.bagua_patch().bagua_pg  # pytype: disable=attribute-error
    elif type(process_group) is BaguaProcessGroup:
        return process_group
    else:
        raise Exception("unexpect input {}".format(type(process_group)))


class DistributedDataParallel_V1_9_0(DistributedDataParallel_V1_9_0_Interface):
    r"""
    `PyTorch v1.9.0 DDP <https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/parallel/distributed.py#L125>`_ compatible interface.
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
        gradient_as_bucket_view=True,
        # The following bagua parameters
        optimizers: List[torch.optim.Optimizer] = [],
        algorithm: "bagua.torch_api.algorithms.Algorithm" = GradientAllReduceAlgorithm(),
    ) -> None:
        """Bagua internal use function. Construction use :class:`DistributedDataParallel`."""
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
        self.find_unused_parameters = find_unused_parameters

        if not hasattr(module, "_bagua_module_name"):
            module._bagua_module_name = "{}_{}".format(
                self.__class__.__name__, id(module)
            )

        self.inner = BaguaDistributedDataParallel(
            self.module,
            optimizers,
            algorithm,
            process_group=to_bagua_process_group(process_group),
            gradient_as_bucket_view=gradient_as_bucket_view,
            find_unused_parameters=find_unused_parameters,
            bagua_module_name=module.bagua_module_name,
        )

    @property
    def require_backward_grad_sync(self):
        """
        DDP gradient synchronizations switch, see :meth:`no_sync` for usage.
        """
        return self.inner.require_backward_grad_sync

    @property
    def parameters_to_ignore(self):
        """Parameters that will be ignored in DDP."""
        return self.inner.parameters_to_ignore

    def forward(self, *inputs, **kwargs):
        output = self.module(*inputs, **kwargs)
        return output

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> ddp = bagua.torch_api.data_parallel.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            >>>   for input in inputs:
            >>>     ddp(input).backward()  # no synchronization, accumulate grads
            >>> ddp(another_input).backward()  # synchronize grads
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.inner.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.inner.require_backward_grad_sync = old_require_backward_grad_sync

    @property
    def bagua_algorithm(self):
        """
        The algorithm implementation used by the module,
        reified by the algorithm passed in from the constructor.
        """
        return self.inner.bagua_algorithm

    @property
    def bagua_optimizers(self):
        """
        Optimizer(s) used by the module. It can contain one or more PyTorch optimizers.
        """
        return self.inner.bagua_optimizers

    @property
    def bagua_buckets(self):
        """
        All Bagua buckets in a list.
        """
        return self.inner.bagua_buckets


def DistributedDataParallel(
    module: Module,
    device_ids: Optional[List[Union[int, torch.device]]] = None,
    output_device: Union[int, torch.device] = None,
    dim: int = 0,
    broadcast_buffers: bool = True,
    process_group: Union[None, TorchProcessGroup] = None,
    bucket_cap_mb: int = 25,
    find_unused_parameters: bool = False,
    check_reduction: bool = False,
    gradient_as_bucket_view: bool = True,
    # The followings are parameters for Bagua
    optimizers: List[torch.optim.Optimizer] = [],
    algorithm: "bagua.torch_api.algorithms.Algorithm" = GradientAllReduceAlgorithm(),
) -> Union[TorchDistributedDataParallel, DistributedDataParallel_V1_9_0]:
    r"""
    This function provides a `PyTorch DDP <https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/parallel/distributed.py#L125>`_ compatible
    interface plus several Bagua specific parameters.

    Args:
        module (Module): module to be parallelized
        device_ids (Optional[List[Union[int, torch.device]]], optional): CUDA devices.

                   1) For single-device modules, ``device_ids`` can
                   contain exactly one device id, which represents the only
                   CUDA device where the input module corresponding to this process resides.
                   Alternatively, ``device_ids`` can also be ``None``.

                   2) For multi-device modules and CPU modules,
                   ``device_ids`` must be ``None``.

                   When ``device_ids`` is ``None`` for both cases,
                   both the input data for the forward pass and the actual module
                   must be placed on the correct device.
                   (default: ``None``)
        output_device (Union[int, torch.device], optional): Device location of output for
                      single-device CUDA modules. For multi-device modules and
                      CPU modules, it must be ``None``, and the module itself
                      dictates the output location. (default: ``device_ids[0]``
                      for single-device modules)
        dim (int, optional): Flag that enables syncing (broadcasting)
                          buffers of the module at beginning of the ``forward``
                          function. (default: ``True``)
        broadcast_buffers (bool, optional): Flag that enables syncing (broadcasting)
                          buffers of the module at beginning of the ``forward``
                          function. (default: ``True``)
        process_group (Union[None, TorchProcessGroup], optional): The process group to be used for distributed data
                       all-reduction. If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
        bucket_cap_mb (int, optional): ``DistributedDataParallel`` will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       :attr:`bucket_cap_mb` controls the bucket size in
                       MegaBytes (MB). (default: 25)
        find_unused_parameters (bool, optional): Traverse the autograd graph from all
                               tensors contained in the return value of the
                               wrapped module's ``forward`` function. Parameters
                               that don't receive gradients as part of this
                               graph are preemptively marked as being ready to
                               be reduced. In addition, parameters that may have
                               been used in the wrapped module's ``forward``
                               function but were not part of loss computation and
                               thus would also not receive gradients are
                               preemptively marked as ready to be reduced.
                               (default: ``False``)
        check_reduction (bool, optional): This argument is deprecated.
        gradient_as_bucket_view (bool, optional): When set to ``True``, gradients will be views
                      pointing to different offsets of ``allreduce`` communication
                      buckets. This can reduce peak memory usage, where the
                      saved memory size will be equal to the total gradients
                      size. Moreover, it avoids the overhead of copying between
                      gradients and ``allreduce`` communication buckets. When
                      gradients are views, ``detach_()`` cannot be called on the
                      gradients. If hitting such errors, please fix it by
                      referring to the :meth:`~torch.optim.Optimizer.zero_grad`
                      function in ``torch/optim/optimizer.py`` as a solution.
        optimizers (List[torch.optim.Optimizer], optional): Optimizer(s) used by the module. It can contain one or more PyTorch optimizers. Defaults to ``[]``.
        algorithm (bagua.torch_api.algorithms.Algorithm, optional): Data
                parallel distributed algorithm, decide how to communication mode
                and the way the model is updated. Defaults to :class:`~bagua.torch_api.algorithms.gradient_allreduce.GradientAllReduceAlgorithm`.

    Returns:
        Union[TorchDistributedDataParallel, DistributedDataParallel_V1_9_0]: Bagua distributed data parallel instance used for distributed training.

    Example::

        >>> bagua.init_process_group()
        >>> net = bagua.data_parallel.DistributedDataParallel(model)

    Example using faster algorithms in Bagua::

        >>> from bagua.torch_api.algorithms import bytegrad
        >>> bagua.init_process_group()
        >>> net = bagua.data_parallel.DistributedDataParallel(model, algorithm=bytegrad.ByteGradAlgorithm())
        >>> # For more possible algorithms, see https://tutorials.baguasys.com/algorithms/.

    """

    check_list = [
        device_ids is None,
        output_device is None,
        dim == 0,
        broadcast_buffers is True,
        check_reduction is False,
    ]
    if not all(check_list):
        warnings.warn(
            "Some parameters passed into BaguaDistributedDataParallel"
            " have not been supported yet. Bagua has automatically "
            "fallback to upstream PyTorch DistributedDataParallel "
            "implementation. If this is unexpected, please submit "
            "an issue to https://github.com/BaguaSys/bagua. Thanks."
        )
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
