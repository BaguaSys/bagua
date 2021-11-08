#!/usr/bin/env python3

from __future__ import annotations
from bagua.torch_api.communication import get_backend, _get_default_group
from typing import List, Callable, Optional

import bagua_core as B
import torch

from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.utils import check_contiguous, get_flattened_tensor
from bagua.torch_api.communication import (
    BaguaProcessGroup,
    _bagua_backend_comm,
)


class BaguaBucket:
    def __init__(
        self, tensors: List[BaguaTensor], name: str, flatten: bool, alignment: int = 1
    ) -> None:
        """
        Create a Bagua bucket with a list of Bagua tensors.

        Args:
            tensors: A list of Bagua tensors to be put in the
                bucket.
            name: The unique name of the bucket.
            flatten: If ``True``, flatten the input tensors so that they are
                contiguous in memory.
            alignment: If :attr:`alignment > 1`, Bagua will create a padding tensor to
                the bucket so that the total number of elements in the bucket divides
                the given alignment.
        """
        self.tensors = tensors
        """
        The Bagua tensors contained in the bucket.
        """
        self.bagua_module_name = tensors[0].bagua_module_name
        for tensor in self.tensors:
            assert (
                self.bagua_module_name == tensor.bagua_module_name
            ), "every tensor in the same bucket should have the same model name"
        self._bagua_backend = get_backend(self.bagua_module_name)
        self.name = name
        """
        The bucket's name.
        """
        self.padding_tensor = None

        if alignment > 1:
            padding = sum(tensor.numel() for tensor in self.tensors) % alignment
            if padding > 0:
                padding = alignment - padding
                # padding tensor must be of name bagua_padding_tensor, so that they are always marked as ready for communication in the backend
                self.padding_tensor = torch.zeros(
                    padding, dtype=self.tensors[0].dtype, device=self.tensors[0].device
                ).ensure_bagua_tensor(
                    "bagua_padding_tensor_bucket_" + name,
                    module_name=self.bagua_module_name,
                )

        self._all_tensors = (
            self.tensors + [self.padding_tensor]
            if self.padding_tensor is not None
            else self.tensors
        )

        self.backend_tensor = None
        self.flatten = flatten
        if self.flatten:
            self._flatten_()
            torch.cuda.empty_cache()

        self.backend_bucket = B.BaguaBucketPy(
            name,
            [tensor.bagua_backend_tensor() for tensor in self._all_tensors],
        )

        for tensor in self._all_tensors:
            tensor._bagua_bucket = self

    def flattened_tensor(self) -> torch.Tensor:
        """
        Returns a tensor contiguous in memory which contains the same data as effective tensors, i.e.
        returned by calling :meth:`~bagua.torch_api.tensor.BaguaTensor.bagua_getter_closure` on
        :attr:`self` tensors and padding tensor (if exists).
        """

        all_effective_tensors = [
            tensor.bagua_getter_closure() for tensor in self._all_tensors
        ]
        return get_flattened_tensor(all_effective_tensors)

    def _flatten_(self):
        """
        Flatten effective tensors in place.
        """
        if len(self._all_tensors) == 0:
            return

        flatten_tensor = self.flattened_tensor()

        if self.check_flatten():
            flatten_tensor.set_(
                self._all_tensors[0].bagua_getter_closure().storage(),
                0,
                flatten_tensor.shape,
            )
            self.backend_tensor = flatten_tensor
            return

        flatten_storage = flatten_tensor.storage()
        offset = 0

        for tensor in self._all_tensors:
            tensor.bagua_set_storage(flatten_storage, offset)
            offset += tensor.bagua_getter_closure().numel()

        # set backend tensor
        self.backend_tensor = flatten_tensor
        # check
        assert self.check_flatten()

    def check_flatten(self) -> bool:
        """
        Returns:
            True if effective tensors are contiguous in memory.
        """
        return check_contiguous(
            [tensor.bagua_getter_closure() for tensor in self._all_tensors]
        )

    def append_python_op(
        self,
        python_function: Callable[[str], None],
        group: Optional[BaguaProcessGroup] = None,
    ):
        """
        Append a Python operation to a bucket. A Python operation is a Python function that
        takes the bucket's name and returns ``None``. It can do arbitrary things within the
        function body.

        The operations will be executed by the Bagua backend in the order they are appended
        when all the tensors within the bucket are marked ready.

        Args:
            python_function: The Python operation function.
            group: The process group to work on. If ``None``, the default process group will be used.
        """

        if group is None:
            group = _get_default_group()

        def wrapper_function_factory(pyop):
            def wrapped_pyop(name):
                with torch.cuda.stream(group.stream):
                    return pyop(name)

            return wrapped_pyop

        self.backend_bucket.append_python_op(wrapper_function_factory(python_function))

    def append_centralized_synchronous_op(
        self,
        hierarchical: bool = False,
        average: bool = True,
        scattergather: bool = False,
        compression: Optional[str] = None,
        group: Optional[BaguaProcessGroup] = None,
    ):
        """
        Append a centralized synchronous operation to a bucket. It will sum or average the tensors in the bucket
        for all workers.

        The operations will be executed by the Bagua backend in the order they are appended
        when all the tensors within the bucket are marked ready.

        Args:
            hierarchical (bool): Enable hierarchical communication. Which means the GPUs on the same machine
                will communicate will each other first. After that, machines do inter-node communication. This can
                boost performance when the inter-node communication cost is high.
            average (bool): If ``True``, the gradients on each worker are averaged. Otherwise, they are summed.
            scattergather (bool): If ``True``, the communication between workers are done with scatter gather instead
                of allreduce. This is required for using compression.
            compression: If not ``None``, the tensors will be compressed for communication. Currently ``"MinMaxUInt8"`` is
                supported.
            group: The process group to work on. If ``None``, the default process group will be used.
        """
        if group is None:
            group = _get_default_group()

        if hierarchical:
            self.backend_bucket.append_centralized_synchronous_op(
                _bagua_backend_comm(group.get_inter_node_communicator()),
                _bagua_backend_comm(group.get_intra_node_communicator()),
                hierarchical=hierarchical,
                average=average,
                scattergather=scattergather,
                compression=compression,
            )
        else:
            self.backend_bucket.append_centralized_synchronous_op(
                _bagua_backend_comm(group.get_global_communicator()),
                None,
                hierarchical=hierarchical,
                average=average,
                scattergather=scattergather,
                compression=compression,
            )

    def append_decentralized_synchronous_op(
        self,
        peer_weight: BaguaTensor,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
        group: Optional[BaguaProcessGroup] = None,
    ):
        """
        Append a decentralized synchronous operation to a bucket. It will do gossipy style model averaging among workers.

        This operation is not inplace, which means the bucket weights is first copied to :attr:`peer_weight`, and the result of
        decentralized averaging will be in :attr:`peer_weight`. To copy :attr:`peer_weight` back to :attr:`self`, call
        ``op.copy_back_peer_weight(self)``.

        This operation will be executed by the Bagua backend in
        the order they are appended when all the tensors within the bucket are marked ready.

        Args:
            peer_weight (BaguaTensor):  A tensor used for averaging model with peers, should be of the same size
                with the bucket tensors total size. Use ``self.flattened_tensor().ensure_bagua_tensor(...)`` to create such a tensor.
            hierarchical (bool): Enable hierarchical communication. Which means the GPUs on the same machine
                will communicate will each other first. After that, machines do inter-node communication. This can
                boost performance when the inter-node communication cost is high.
            peer_selection_mode (str): Can be ``"all"`` or ``"shift_one"``. ``"all"`` means all workers' weights are averaged
                in each communication step. ``"shift_one"`` means each worker selects a different peer to do weights average
                in each communication step.
            group: The process group to work on. If ``None``, the default process group will be used.
        Returns:
            The decentralized synchronous operation itself.
        """
        if group is None:
            group = _get_default_group()

        if hierarchical:
            return self.backend_bucket.append_decentralized_synchronous_op(
                _bagua_backend_comm(group.get_inter_node_communicator()),
                _bagua_backend_comm(group.get_intra_node_communicator()),
                hierarchical=hierarchical,
                peer_selection_mode=peer_selection_mode,
                peer_weight=peer_weight.bagua_backend_tensor(),
            )
        else:
            return self.backend_bucket.append_decentralized_synchronous_op(
                _bagua_backend_comm(group.get_global_communicator()),
                None,
                hierarchical=hierarchical,
                peer_selection_mode=peer_selection_mode,
                peer_weight=peer_weight.bagua_backend_tensor(),
            )

    def append_low_precision_decentralized_synchronous_op(
        self,
        weight: BaguaTensor,
        left_peer_weight: BaguaTensor,
        right_peer_weight: BaguaTensor,
        hierarchical: bool = True,
        compression: str = "MinMaxUInt8",
        group: Optional[BaguaProcessGroup] = None,
    ):
        """
        Append a low precision decentralized synchronous operation to a bucket. It will compress the difference
        of local models between two successive iterations and exchange them among workers.

        The operations will be executed by the Bagua backend in the order they are appended
        when all the tensors within the bucket are marked ready.

        Args:
            weight (BaguaTensor): Model replica of current worker's local model. It should be of the same size
                with the bucket tensors total size. Use ``self.flattened_tensor().ensure_bagua_tensor(...)`` to create such a tensor.
            left_peer_weight (BaguaTensor): Model replica of current worker's left peer. It should be of the same size
                with the bucket tensors total size. Use ``self.flattened_tensor().ensure_bagua_tensor(...)`` to create such a tensor,
                then copy the initializing weights of current worker's left peer to the tensor.
            right_peer_weight (BaguaTensor): Model replica of current worker's right peer. It should be of the same size
                with the bucket tensors total size. Use ``self.flattened_tensor().ensure_bagua_tensor(...)`` to create such a tensor.
                then copy the initializing weights of current worker's right peer to the tensor.
            hierarchical (bool): Enable hierarchical communication. Which means the GPUs on the same machine
                will communicate will each other first. After that, machines do inter-node communication. This can
                boost performance when the inter-node communication cost is high.
            compression (str): The way how tensors are compressed for communication. Currently ``"MinMaxUInt8"`` is supported.
            group: The process group to work on. If ``None``, the default process group will be used.
        """
        if group is None:
            group = _get_default_group()

        if hierarchical:
            self.backend_bucket.append_low_precision_decentralized_synchronous_op(
                _bagua_backend_comm(group.get_inter_node_communicator()),
                _bagua_backend_comm(group.get_intra_node_communicator()),
                hierarchical=hierarchical,
                peer_selection_mode="ring",
                compression=compression,
                weight=weight.bagua_backend_tensor(),
                left_peer_weight=left_peer_weight.bagua_backend_tensor(),
                right_peer_weight=right_peer_weight.bagua_backend_tensor(),
            )
        else:
            self.backend_bucket.append_low_precision_decentralized_synchronous_op(
                _bagua_backend_comm(group.get_global_communicator()),
                None,
                hierarchical=hierarchical,
                peer_selection_mode="ring",
                compression=compression,
                weight=weight.bagua_backend_tensor(),
                left_peer_weight=left_peer_weight.bagua_backend_tensor(),
                right_peer_weight=right_peer_weight.bagua_backend_tensor(),
            )

    def append_asynchronous_model_average_op(
        self, peer_selection_mode: str, group: Optional[BaguaProcessGroup] = None
    ):

        """
        Append an asynchronous model average operation to a bucket. This operation will enable continuous
        model averaging between workers while training a model.

        The operations will be executed by the Bagua backend in the order they are appended
        when all the tensors within the bucket are marked ready.

        This operation is intended to run in parallel with the computation process. It returns a reference
        to the op. The op features a lock to exclusively access the model. Call ``op.lock_weight()`` to
        acquire the lock and ``op.unlock_weight()`` to release it.

        Args:
            peer_selection_mode (str): The way how workers communicate with each otehr. Currently ``"all"`` is supported.
                ``"all"`` means all workers' weights are averaged during each communication.
            group: The process group to work on. If ``None``, the default process group will be used.
        Returns:
            The asynchronous model average operation itself.
        """
        if group is None:
            group = _get_default_group()

        return self.backend_bucket.append_decentralized_asynchronous_op(
            _bagua_backend_comm(group.get_global_communicator()),
            None,
            peer_selection_mode=peer_selection_mode,
            torch_stream=torch.cuda.current_stream().cuda_stream,
        )

    def clear_ops(self) -> BaguaBucket:
        """
        Clear the previously appended operations.
        """
        self.backend_bucket.clear_ops()
        return self

    def bytes(self) -> int:
        """Returns the total number of bytes occupied by the bucket."""
        effective_tensors = [tensor.bagua_getter_closure() for tensor in self.tensors]
        return sum(
            tensor.numel() * tensor.element_size() for tensor in effective_tensors
        )
