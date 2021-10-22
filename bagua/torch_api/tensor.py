#!/usr/bin/env python3
from bagua.torch_api.communication import get_backend
from typing import Optional, Callable

import torch
import bagua_core as B
import gorilla


@gorilla.patches(torch.Tensor, filter=lambda name, obj: "bagua" in name)
class BaguaTensor:
    """
    This class patch `torch.Tensor <https://pytorch.org/docs/stable/tensors.html?highlight=tensor#torch.Tensor>`_
    with additional methods.
    """

    def _bagua_sanity_check(self):
        assert (
            self._bagua_backend_tensor.data_ptr()
            == self.bagua_getter_closure().data_ptr()
        )
        assert (
            self._bagua_backend_tensor.num_elements()
            == self.bagua_getter_closure().numel()
        )
        assert (
            self._bagua_backend_tensor.num_elements_allocated()
            == self.bagua_getter_closure().numel()
        )

    def is_bagua_tensor(self) -> bool:
        """
        Checking if this is a Bagua tensor.
        """
        return hasattr(self, "_bagua_backend_tensor")

    def ensure_bagua_tensor(
        self,
        name: Optional[str] = None,
        module_name: Optional[str] = None,
        getter_closure: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        setter_closure: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Convert a PyTorch tensor or parameter to Bagua tensor inplace and return it.

        This operation will register a tensor to the Bagua backend, which is required to use
        Bagua's communication algorithms. The tensor can be get and set by calling
        :attr:`getter_closure` and :attr:`setter_closure` on `self` respectively.

        Args:
            name: The unique name of the tensor.
            module_name: The name of the model of which the tensor belongs to.
              The model name can be acquired using ``model.bagua_module_name``.
              This is required to call :meth:`bagua_mark_communication_ready` related methods.
            getter_closure: A function who accepts `self` as its input and returns a tensor as
              its output. It is used to retrieve the tensor to be registered to the Bagua backend.
              If ``None``, register `self`. Default: ``None``.
            setter_closure: A function who accepts `self` and another tensor as its inputs. Used to
              reset the tensor registered. If ``None``, it's a no-op. Default: ``None``.

        Returns:
            The original tensor with Bagua tensor attributes initialized.
        """
        if self.is_bagua_tensor():
            if name is not None:
                assert (
                    self.bagua_tensor_name == name
                ), "assigning a different name to existing bagua tensor is forbidden"

                assert (
                    self.bagua_module_name == module_name
                ), "assigning a different module name to existing bagua tensor is forbidden"

            if (
                getter_closure == self._bagua_getter_closure
                and setter_closure == self._bagua_setter_closure
            ):
                return self

        self.bagua_tensor_name = name if name is not None else ""
        self.bagua_module_name = module_name
        self.bagua_backend = (
            get_backend(self.bagua_module_name)
            if self.bagua_module_name is not None
            else None
        )

        # initialize backend tensor
        if setter_closure is not None:
            assert (
                getter_closure is not None
            ), "must provide `getter_closure` when `setter_closure` is not None"
        self._bagua_getter_closure = getter_closure
        self._bagua_setter_closure = setter_closure

        self._bagua_backend_tensor = B.BaguaTensorPy(
            name=self.bagua_tensor_name,
            torch_tensor=self,
        )

        self._bagua_sanity_check()

        self._bagua_ready_event = torch.cuda.Event()
        self._bagua_bucket = None
        return self

    def bagua_getter_closure(self) -> torch.Tensor:
        """Returns the tensor registered."""
        return (
            self._bagua_getter_closure(self)
            if self._bagua_getter_closure is not None
            else self
        )

    def bagua_setter_closure(self, tensor: torch.Tensor):
        """Sets the tensor registered to :attr:`tensor`."""

        assert self._bagua_setter_closure is not None
        self._bagua_setter_closure(self, tensor)

    def to_bagua_tensor(
        self,
        name: Optional[str] = None,
        module_name: Optional[str] = None,
        getter_closure: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        setter_closure: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Create a new Bagua tensor from a PyTorch tensor or parameter and return it.
        The original tensor is not changed. A Bagua tensor is required to use
        Bagua's communication algorithms. See :meth:`ensure_bagua_tensor` for more
        information.

        Args:
            name: The unique name of the tensor.
            module_name: The name of the model of which the tensor belongs to.
              The model name can be acquired using ``model.bagua_module_name``.
              This is required to call :meth:`bagua_mark_communication_ready` related methods.
            getter_closure: A function to retrieve the tensor to be registered to the Bagua backend.
              See :meth:`ensure_bagua_tensor`.
            setter_closure: A function to reset the registered tensor. See :meth:`ensure_bagua_tensor`.

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

    def bagua_set_registered_storage(
        self,
        storage: torch.Storage,
        storage_offset: int = 0,
    ):
        """
        Sets the underlying storage for the tensor registered using an existing
        `torch.Storage <https://pytorch.org/docs/stable/storage.html?highlight=storage>`_.

        Args:
            storage: The storage to use.
            storage_offset: The offset in the storage.
        """
        if self._bagua_setter_closure is None:
            # set directly
            with torch.no_grad():
                self.bagua_getter_closure().set_(storage, storage_offset, self.shape)
            return

        with torch.no_grad():
            t = torch.zeros_like(self.bagua_getter_closure())
            t.set_(storage, storage_offset, t.shape)
            self.bagua_setter_closure(t)


_base = gorilla._get_base(BaguaTensor)
_decorator_data = gorilla.get_decorator_data(_base)
for patch in _decorator_data.patches:
    gorilla.apply(patch)
