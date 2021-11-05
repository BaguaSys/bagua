#!/usr/bin/env python3
from bagua.torch_api.communication import get_backend
from typing import Optional, Callable

import torch
import bagua_core as B
import gorilla
from setuptools import distutils

LooseVersion = distutils.version.LooseVersion


@gorilla.patches(torch.Tensor, filter=lambda name, obj: "bagua" in name)
class BaguaTensor:
    """
    This class patch `torch.Tensor <https://pytorch.org/docs/stable/tensors.html?highlight=tensor#torch.Tensor>`_
    with additional methods.

    A Bagua tensor is required to use Bagua's communication algorithms. Users can convert a PyTorch tensor to Bagua
    tensor by :meth:`ensure_bagua_tensor`.

    Bagua tensor features a proxy structure, where the actual tensor used by backend is accessed via a **"Proxy Tensor"**.
    The proxy tensor is registered in Bagua, whenever the Bagua backend needs a tensor (for example use it for
    communication), it calls the :meth:`bagua_getter_closure` on the proxy tensor to get the tensor that is actually
    worked on. We call this tensor **"Effective Tensor"**. The :attr:`bagua_setter_closure` is also provided to replace
    the effective tensor during runtime. It is intended to be used to replace the effective tensor with customized
    workflow.

    Their relation can be seen in the following diagram:

    .. image:: https://user-images.githubusercontent.com/18649508/139179394-51d0c0f5-e233-4ada-8e5e-0e70a889540d.png

    For example, in the gradient allreduce algorithm, the effective tensor that needs to be exchanged between
    machines is the gradient.  In this case, we will register the model parameters as proxy tensor, and register
    :meth:`bagua_getter_closure` to be ``lambda proxy_tensor: proxy_tensor.grad``. In this way, even if the gradient
    tensor is recreated or changed during runtime, Bagua can still identify the correct tensor and use it
    for communication, since the proxy tensor serves as the root for access and is never replaced.
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
        A Bagua tensor is required to use Bagua's communication algorithms.

        This operation will register :attr:`self` as proxy tensor to the Bagua backend. :attr:`getter_closure` takes
        the proxy tensor as input and returns a Pytorch tensor. When using the Bagua tensor, the :attr:`getter_closure`
        will be called and returns the effective tensor which will be used for communication and other operations.
        For example, if one of a model's parameter ``param`` is registered as proxy tensor, and :attr:`getter_closure`
        is ``lambda x: x.grad``, during runtime its gradient will be used.

        :attr:`setter_closure` takes the proxy tensor and another tensor as inputs and returns nothing.
        It is mainly used for changing the effective tensor used in runtime. For example when one of
        a model's parameter ``param`` is registered as proxy tensor, and :attr:`getter_closure` is ``lambda x: x.grad``,
        the :attr:`setter_closure` can be ``lambda param, new_grad_tensor: setattr(param, "grad", new_grad_tensor)``.
        When the :attr:`setter_closure` is called, the effective tensor used in later operations will be changed
        to ``new_grad_tensor``.

        Args:
            name: The unique name of the tensor.
            module_name: The name of the model of which the tensor belongs to.
              The model name can be acquired using ``model.bagua_module_name``.
              This is required to call :meth:`bagua_mark_communication_ready` related methods.
            getter_closure: A function that accepts a Pytorch tensor as its input and returns a Pytorch tensor as
              its output. Could be ``None``, which means an identity mapping ``lambda x: x`` is used. Default: ``None``.
            setter_closure: A function that accepts two Pytorch tensors as its inputs and returns nothing. Could be ``None``,
              which is a no-op. Default: ``None``.

        Returns:
            The original tensor with Bagua tensor attributes initialized.
        """
        if self.is_bagua_tensor():
            if name is not None:
                assert (
                    self.bagua_tensor_name == name
                ), "assigning a different name to existing bagua tensor is forbidden"
                ", self.bagua_tensor_name={}, name={}".format(self.bagua_tensor_name, name)

                assert (
                    self.bagua_module_name == module_name
                ), "assigning a different module name to existing bagua tensor is forbidden"
                ", self.bagua_module_name={}, module_name={}".format(self.bagua_module_name, module_name)

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

    def to_bagua_tensor(
        self,
        name: Optional[str] = None,
        module_name: Optional[str] = None,
        getter_closure: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        setter_closure: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Create a new Bagua tensor from a PyTorch tensor or parameter and return it.
        The new Bagua tensor will share the same storage with the input PyTorch tensor.
        A Bagua tensor is required to use Bagua's communication algorithms.
        See :meth:`ensure_bagua_tensor` for more information.

        Caveat: Be aware that if the original tensor changes to use a different storage
        using for example ``torch.Tensor.set_(...)``, the new Bagua tensor will still
        use the old storage.

        Args:
            name: The unique name of the tensor.
            module_name: The name of the model of which the tensor belongs to.
              The model name can be acquired using ``model.bagua_module_name``.
              This is required to call :meth:`bagua_mark_communication_ready` related methods.
            getter_closure: A function that accepts a Pytorch tensor as its input and returns a Pytorch tensor as
              its output. See :meth:`ensure_bagua_tensor`.
            setter_closure: A function that accepts two Pytorch tensors as its inputs and returns nothing. See :meth:`ensure_bagua_tensor`.
        Returns:
            The new Bagua tensor sharing the same storage with the original tensor.
        """
        if LooseVersion(torch.__version__) >= LooseVersion("1.10"):
            new_tensor = self.view(self.dtype)
        else:
            new_tensor = torch.Tensor(cdata=self._cdata)
        return new_tensor.ensure_bagua_tensor(
            name, module_name, getter_closure, setter_closure
        )

    def bagua_getter_closure(self) -> torch.Tensor:
        """Returns the tensor that will be used in runtime."""
        return (
            self._bagua_getter_closure(self)
            if self._bagua_getter_closure is not None
            else self
        )

    def bagua_setter_closure(self, tensor: torch.Tensor):
        """Sets the tensor that will be used in runtime to a new Pytorch tensor :attr:`tensor`.

        Args:
            tensor: The new tensor to be set to.
        """

        assert self._bagua_setter_closure is not None
        self._bagua_setter_closure(self, tensor)

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
        Sets the underlying storage for the effective tensor returned by :meth:`bagua_getter_closure` with an existing
        `torch.Storage <https://pytorch.org/docs/stable/storage.html?highlight=storage>`_.

        Args:
            storage: The storage to use.
            storage_offset: The offset in the storage.
        """
        if self._bagua_setter_closure is None:
            # set directly
            with torch.no_grad():
                self.bagua_getter_closure().set_(
                    storage, storage_offset, self.bagua_getter_closure().shape
                )
            return

        with torch.no_grad():
            t = torch.zeros_like(self.bagua_getter_closure())
            t.set_(storage, storage_offset, t.shape)
            self.bagua_setter_closure(t)


_base = gorilla._get_base(BaguaTensor)
_decorator_data = gorilla.get_decorator_data(_base)
for patch in _decorator_data.patches:
    gorilla.apply(patch)
