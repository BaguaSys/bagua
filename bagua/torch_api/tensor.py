#!/usr/bin/env python3
from typing import Optional

from bagua.torch_api.globals import _get_global_state
import torch
import bagua_core as B
import gorilla


@gorilla.patches(torch.Tensor, filter=lambda name, obj: "bagua" in name)
class BaguaTensor:
    """
    This class patch torch.Tensor with additional methods.
    """

    def _bagua_sanity_check(self):
        assert self._bagua_backend_tensor.data_ptr() == self.data_ptr()
        assert self._bagua_backend_tensor.num_elements() == self.numel()
        assert self._bagua_backend_tensor.num_elements_allocated() == self.numel()

    def is_bagua_tensor(self) -> bool:
        return hasattr(self, "_bagua_backend_tensor")

    def ensure_bagua_tensor(self, name: Optional[str] = None):
        """
        Convert a PyTorch tensor or parameter to Bagua tensor inplace and return it.
        A Bagua tensor is required to use Bagua's communication algorithms.

        Args:
            name: the unique name of the tensor

        Returns:
            The original tensor with Bagua tensor attributes initialized.
        """
        if self.is_bagua_tensor():
            if name is not None:
                assert (
                    self.bagua_tensor_name == name
                ), "assigning a different name to existing bagua tensor is forbidden"
            return
        self.bagua_tensor_name = name if name is not None else ""
        self._bagua_backend_tensor = B.BaguaTensorPy(
            name=self.bagua_tensor_name,
            torch_tensor=self,
        )
        self._bagua_sanity_check()
        self._bagua_backend = _get_global_state().get_backend()
        self._bagua_ready_event = torch.cuda.Event()
        self._bagua_bucket = None
        return self

    def to_bagua_tensor(self, name: Optional[str] = None):
        """
        Create a new Bagua tensor from a PyTorch tensor or parameter and return it.
        The original tensor is not changed. A Bagua tensor is required to use
        Bagua's communication algorithms.

        Args:
            name: the unique name of the tensor

        Returns:
            The new Bagua tensor sharing the same storage with the original tensor.
        """
        new_tensor = torch.Tensor(cdata=self._cdata)
        return new_tensor.ensure_bagua_tensor(name)

    def bagua_backend_tensor(self) -> B.BaguaTensorPy:
        """
        Returns:
            The raw Bagua backend tensor.
        """
        return self._bagua_backend_tensor

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
            raise NotImplemented

    def bagua_mark_communication_ready(self):
        """
        Mark a Bagua tensor ready for scheduled operations execution.
        """
        torch.cuda.current_stream().record_event(self._bagua_ready_event)
        self._bagua_backend.mark_communication_ready(
            self._bagua_backend_tensor,
            self._bagua_ready_event.cuda_event,
        )

    def bagua_mark_communication_ready_without_synchronization(self):
        """
        Mark a Bagua tensor ready immediately, without CUDA event synchronization.
        """
        self._bagua_backend.mark_communication_ready(
            self._bagua_backend_tensor,
            0,
        )

    def bagua_set_storage(self, storage: torch.Storage, storage_offset: int = 0):
        """
        Sets the underlying storage using an existing torch.Storage.

        Args:
            storage: the storage to use
            storage_offset: the offset in the storage
        """
        with torch.no_grad():
            self.set_(storage, storage_offset, self.shape)


base = gorilla._get_base(BaguaTensor)
decorator_data = gorilla.get_decorator_data(base)
for patch in decorator_data.patches:
    gorilla.apply(patch)
