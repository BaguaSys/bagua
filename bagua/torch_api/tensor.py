#!/usr/bin/env python3
from bagua.torch_api.communication import get_backend
from typing import Optional

import torch
import bagua_core as B
import gorilla


class BaguaTensor:
    def numel(self):
        return self._tensor.numel()

    def dtype(self):
        return self._tensor.dtype

    def data_ptr(self):
        return self._tensor.data_ptr()

    def device(self):
        return self._tensor.device

    def data(self):
        return self._tensor.data

    def element_size(self):
        return self._tensor.element_size()

    def _bagua_sanity_check(self):
        assert self._bagua_backend_tensor.data_ptr() == self._tensor.data_ptr()
        assert self._bagua_backend_tensor.num_elements() == self._tensor.numel()
        assert (
            self._bagua_backend_tensor.num_elements_allocated() == self._tensor.numel()
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        name: Optional[str] = None,
        module_name: Optional[str] = None,
    ):
        """
        A Bagua tensor is required to use Bagua's communication algorithms.

        Args:
            tensor: A Pytorch Tensor.
            name: The unique name of the tensor.
            module_name: The name of the model of which the tensor belongs to.
              The model name can be acquired using ``model.bagua_module_name``.
              This is required to call :meth:`bagua_mark_communication_ready` related methods.
        """
        self._tensor = tensor

        self.bagua_tensor_name = name if name is not None else ""
        self.bagua_module_name = module_name
        self.bagua_backend = (
            get_backend(self.bagua_module_name)
            if self.bagua_module_name is not None
            else None
        )
        self._bagua_backend_tensor = B.BaguaTensorPy(
            name=self.bagua_tensor_name,
            torch_tensor=self._tensor,
        )
        self._bagua_sanity_check()
        self._bagua_ready_event = torch.cuda.Event()
        self._bagua_bucket = None

    def bagua_backend_tensor(self) -> B.BaguaTensorPy:
        """
        Returns:
            The raw Bagua backend tensor.
        """
        return self._bagua_backend_tensor

    def bagua_mark_communication_ready(self):
        """
        Mark a Bagua tensor ready for scheduled operations execution.
        """
        torch.cuda.current_stream().record_event(self._bagua_ready_event)
        assert (
            self.bagua_backend is not None
        ), "tensor must be initialized with module name to call mark ready"
        self.bagua_backend.mark_communication_ready(
            self._bagua_backend_tensor,
            self._bagua_ready_event.cuda_event,
        )

    def bagua_mark_communication_ready_without_synchronization(self):
        """
        Mark a Bagua tensor ready immediately, without `CUDA event <https://pytorch.org/docs/stable/generated/torch.cuda.Event.html?highlight=event#torch.cuda.Event>`_ synchronization.
        """
        assert (
            self.bagua_backend is not None
        ), "tensor must be initialized with module name to call mark ready"
        self.bagua_backend.mark_communication_ready(
            self._bagua_backend_tensor,
            0,
        )

    def reset_(self, tensor: torch.Tensor):
        """
        Set the underlying tensor in-place.
        """
        self._tensor = tensor
        self._bagua_backend_tensor.reset(tensor)


@gorilla.patches(torch.Tensor, filter=lambda name, obj: "bagua" in name)
class _TorchTensor:
    """
    This class patch `torch.Tensor <https://pytorch.org/docs/stable/tensors.html?highlight=tensor#torch.Tensor>`_ with additional methods.
    """

    def to_bagua_tensor(
        self, name: Optional[str] = None, module_name: Optional[str] = None
    ):
        """
        Create a new Bagua tensor from a PyTorch tensor or parameter and return it.
        The original tensor is not changed. A Bagua tensor is required to use
        Bagua's communication algorithms.

        Args:
            name: The unique name of the tensor.
            module_name: The name of the model of which the tensor belongs to.
              The model name can be acquired using ``model.bagua_module_name``.
              This is required to call :meth:`bagua_mark_communication_ready` related methods.

        Returns:
            The new Bagua tensor sharing the same storage with the original tensor.
        """
        return BaguaTensor(self, name, module_name)

    def bagua_ensure_grad(self) -> torch.Tensor:
        """
        Return the gradient of current parameter. Create a zero gradient tensor
        if not exist.
        """
        if hasattr(self, "grad") and self.grad is not None:
            return self.grad
        elif isinstance(self, torch.nn.Parameter):
            with torch.no_grad():
                t = torch.zeros_like(self.data)
                self.grad = t
            return self.grad
        else:
            raise NotImplementedError


_base = gorilla._get_base(_TorchTensor)
_decorator_data = gorilla.get_decorator_data(_base)
for patch in _decorator_data.patches:
    gorilla.apply(patch)
