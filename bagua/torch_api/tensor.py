#!/usr/bin/env python3
from bagua.torch_api.communication import get_backend
from typing import Optional

import torch
import bagua_core as B
import gorilla


@gorilla.patches(torch.Tensor, filter=lambda name, obj: "bagua" in name)
class BaguaTensor:
    """
    This class patch `torch.Tensor <https://pytorch.org/docs/stable/tensors.html?highlight=tensor#torch.Tensor>`_ with additional methods.
    """

    def _bagua_sanity_check(self):
        assert (
            self._bagua_backend_tensor.data_ptr()
            == self._bagua_getter_closure().data_ptr()
        )
        assert (
            self._bagua_backend_tensor.num_elements()
            == self._bagua_getter_closure().numel()
        )
        assert (
            self._bagua_backend_tensor.num_elements_allocated()
            == self._bagua_getter_closure().numel()
        )

    def is_bagua_tensor(self) -> bool:
        return hasattr(self, "_bagua_backend_tensor")

    def ensure_bagua_tensor(
        self,
        name: Optional[str] = None,
        module_name: Optional[str] = None,
        getter_closure=None,
        setter_closure=None,
    ):
        """
        Convert a PyTorch tensor or parameter to Bagua tensor inplace and return it.
        A Bagua tensor is required to use Bagua's communication algorithms.

        Args:
            name: The unique name of the tensor.
            module_name: The name of the model of which the tensor belongs to.
              The model name can be acquired using ``model.bagua_module_name``.
              This is required to call :meth:`bagua_mark_communication_ready` related methods.

        Returns:
            The original tensor with Bagua tensor attributes initialized.
        """
        if self.is_bagua_tensor():
            if name is not None:
                assert (
                    self.bagua_tensor_name == name
                ), "assigning a different name to existing bagua tensor is forbidden"

        self.bagua_tensor_name = name if name is not None else ""
        self.bagua_module_name = module_name
        self.bagua_backend = (
            get_backend(self.bagua_module_name)
            if self.bagua_module_name is not None
            else None
        )

        # initialize backend tensor
        if setter_closure is not None:
            self._bagua_setter_closure = lambda t: setter_closure(self, t)
            assert (
                getter_closure is not None
            ), "must provide `getter_closure` when `setter_closure` is not None"
        else:
            self._bagua_setter_closure = None

        if getter_closure is not None:
            self._bagua_getter_closure = lambda: getter_closure(self)
        else:
            self._bagua_getter_closure = lambda: self

        self._bagua_backend_tensor = B.BaguaTensorPy(
            name=self.bagua_tensor_name,
            torch_tensor=self,
            python_fallback=(getter_closure is not None),
        )

        self._bagua_ready_event = torch.cuda.Event()
        self._bagua_bucket = None
        return self

    def to_bagua_tensor(
        self,
        name: Optional[str] = None,
        module_name: Optional[str] = None,
        getter_closure=None,
        setter_closure=None,
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
        new_tensor = torch.Tensor(cdata=self._cdata)
        return new_tensor.ensure_bagua_tensor(
            name, module_name, getter_closure, setter_closure
        )

    def bagua_backend_tensor(self) -> B.BaguaTensorPy:
        """
        Returns:
            The raw Bagua backend tensor.
        """
        return self._bagua_backend_tensor

    def bagua_ensure_grad(self) -> torch.Tensor:
        """
        Create a zero gradient tensor for the current parameter if not exist.

        Returns:
            The original tensor.
        """
        if hasattr(self, "grad") and self.grad is not None:
            return self
        elif isinstance(self, torch.nn.Parameter):
            with torch.no_grad():
                t = torch.zeros_like(self.data)
                self.grad = t
            return self
        else:
            raise NotImplementedError

    def bagua_mark_communication_ready(self):
        """
        Mark a Bagua tensor ready for scheduled operations execution.
        """
        torch.cuda.current_stream().record_event(self._bagua_ready_event)
        assert (
            self.bagua_backend is not None
        ), "tensor must be initialized with module name to call mark ready"
        self.bagua_backend.mark_communication_ready(
            self.bagua_backend_tensor(),
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
            self.bagua_backend_tensor(),
            0,
        )

    def bagua_set_storage(
        self,
        storage: torch.Storage,
        storage_offset: int = 0,
    ):
        """
        Sets the underlying storage using an existing `torch.Storage <https://pytorch.org/docs/stable/storage.html?highlight=storage>`_.

        Args:
            storage: The storage to use.
            storage_offset: The offset in the storage.
        """
        if self._bagua_setter_closure is None:
            # set directly
            with torch.no_grad():
                self._bagua_getter_closure().set_(storage, storage_offset, self.shape)
            return

        with torch.no_grad():
            t = torch.zeros_like(self._bagua_getter_closure())
            t.set_(storage, storage_offset, t.shape)
            self._bagua_setter_closure(t)


_base = gorilla._get_base(BaguaTensor)
_decorator_data = gorilla.get_decorator_data(_base)
for patch in _decorator_data.patches:
    gorilla.apply(patch)
