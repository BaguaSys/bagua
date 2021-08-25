#!/usr/bin/env python3

from __future__ import annotations
from bagua.torch_api.communication import get_backend
from typing import List, Callable, Optional

import bagua_core as B
import torch

from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.utils import check_contiguous
from bagua.torch_api.communication import broadcast


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
            alignment: If `alignment > 1`, Bagua will create a padding tensor to
                the bucket so that the total number of elements in the bucket divides
                the given alignment.
        """
        self.tensors = tensors
        """
        The tensors contained within the bucket.
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
                ).to_bagua_tensor("bagua_padding_tensor_bucket_" + name)

        self._all_tensors = (
            self.tensors + [self.padding_tensor]
            if self.padding_tensor is not None
            else self.tensors
        )

        self.backend_tensor = None
        self.flatten = flatten
        if self.flatten:
            self._flatten_()

        self.backend_bucket = B.BaguaBucketPy(
            name, [tensor._bagua_backend_tensor for tensor in self._all_tensors]
        )

        for tensor in self._all_tensors:
            tensor._bagua_bucket = self

    def flattened_tensor(self) -> BaguaTensor:
        """
        Returns a tensor contiguous in memory which contains the same data as `self` tensors and padding tensor (if exists).
        """

        total_size = 0
        for tensor in self._all_tensors:
            total_size += tensor.numel()

        flatten_tensor = torch.zeros(total_size, dtype=self._all_tensors[0].dtype).to(
            self._all_tensors[0].device
        )

        offset = 0
        for tensor in self._all_tensors:
            # copy data
            flatten_tensor[offset : offset + tensor.numel()] = tensor.data.reshape(-1)
            offset += tensor.numel()
        return flatten_tensor

    def _flatten_(self):
        """
        Flatten inner tensors in place.
        """
        if self.check_flatten():
            return

        if len(self._all_tensors) == 0:
            return
        total_size = 0
        for tensor in self._all_tensors:
            total_size += tensor.numel()

        flatten_tensor = torch.zeros(total_size, dtype=self._all_tensors[0].dtype).to(
            self._all_tensors[0].device
        )
        flatten_storage = flatten_tensor.storage()

        offset = 0
        for tensor in self._all_tensors:
            # copy data
            flatten_tensor[offset : offset + tensor.numel()] = tensor.data.reshape(-1)
            tensor.bagua_set_storage(flatten_storage, offset)
            offset += tensor.numel()

        # set backend tensor
        self.backend_tensor = flatten_tensor
        # check
        assert self.check_flatten()

    def check_flatten(self) -> bool:
        """
        Returns:
            True if the bucket's tensors are contiguous in memory.
        """
        return check_contiguous(self._all_tensors)

    def append_python_op(self, python_function: Callable[[str], None]) -> BaguaBucket:
        """
        Append a Python operation to a bucket. A Python operation is a Python function that
        takes the bucket's name and returns ``None``. It can do arbitrary things within the
        function body.

        The operations will be executed by the Bagua backend in the order they are appended
        when all the tensors within the bucket are marked ready.

        Args:
            python_function: The Python operation function.

        Returns:
            The bucket itself.
        """

        def wrapper_function_factory(pyop):
            def wrapped_pyop(name):
                with torch.cuda.stream(self._bagua_backend.stream):
                    return pyop(name)

            return wrapped_pyop

        self.backend_bucket.append_python_op(wrapper_function_factory(python_function))
        return self

    def append_centralized_synchronous_op(
        self,
        hierarchical: bool = False,
        average: bool = True,
        scattergather: bool = False,
        compression: Optional[str] = None,
    ) -> BaguaBucket:
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
            compression: If not ``None``, the tensors will be compressed for communication. Currently "MinMaxUInt8" is
                supported.

        Returns:
            The bucket itself.
        """
        if hierarchical:
            self.backend_bucket.append_centralized_synchronous_op(
                self._bagua_backend.internode_communicator,
                self._bagua_backend.intranode_communicator,
                hierarchical=hierarchical,
                average=average,
                scattergather=scattergather,
                compression=compression,
            )
        else:
            self.backend_bucket.append_centralized_synchronous_op(
                self._bagua_backend.global_communicator,
                None,
                hierarchical=hierarchical,
                average=average,
                scattergather=scattergather,
                compression=compression,
            )
        return self

    def append_decentralized_synchronous_op(
        self,
        peer_weight: BaguaTensor,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
    ) -> BaguaBucket:
        """
        Append a decentralized synchronous operation to a bucket. It will do gossipy style model averaging among workers.

        This operation is not inplace, which means the bucket weights is first copied to `peer_weight`, and the result of
        decentralized averaging will be in `peer_weight`. To copy `peer_weight` back to `self`, call
        :func:`decentralized_synchronous_op_copy_back_peer_weight`.

        This operation will be executed by the Bagua backend in
        the order they are appended when all the tensors within the bucket are marked ready.

        Args:
            peer_weight (BaguaTensor):  A tensor used for averaging model with peers, should be of the same size
                with the bucket tensors total size. Use ``self.flattened_tensor().to_bagua_tensor(...)`` to create such a tensor.
            hierarchical (bool): Enable hierarchical communication. Which means the GPUs on the same machine
                will communicate will each other first. After that, machines do inter-node communication. This can
                boost performance when the inter-node communication cost is high.
            peer_selection_mode (str): Can be "all" or "shift_one". "all" means all workers' weights are averaged
                in each communication step. "shift_one" means each worker selects a different peer to do weights average
                in each communication step.
        Returns:
            The bucket itself.
        """

        if hierarchical:
            self.backend_bucket.append_decentralized_synchronous_op(
                self._bagua_backend.internode_communicator,
                self._bagua_backend.intranode_communicator,
                hierarchical=hierarchical,
                peer_selection_mode=peer_selection_mode,
                peer_weight=peer_weight._bagua_backend_tensor,
            )
        else:
            self.backend_bucket.append_decentralized_synchronous_op(
                self._bagua_backend.global_communicator,
                None,
                hierarchical=hierarchical,
                peer_selection_mode=peer_selection_mode,
                peer_weight=peer_weight._bagua_backend_tensor,
            )
        return self

    def decentralized_synchronous_op_copy_back_peer_weight(
        self, peer_weight: BaguaTensor, hierarchical: bool = True
    ):
        """
        Copy `peer_weight` back to bucket weights to end a decentralized synchronous operation.
        See :func:`append_decentralized_synchronous_op` for more information.

        Args:
            peer_weight (BaguaTensor):  A tensor used for averaging model with peers, should be of the same size
                with the bucket tensors total size. Use ``self.flattened_tensor().to_bagua_tensor(...)`` to create such a tensor.
            hierarchical (bool): Enable hierarchical communication. Which means the GPUs on the same machine
                will communicate will each other first. After that, machines do inter-node communication. This can
                boost performance when the inter-node communication cost is high. Must be the same with `hierarchical` argument in
                :func:`append_decentralized_synchronous_op`.
        """
        intra_comm = self._bagua_backend.intranode_communicator
        inter_comm = self._bagua_backend.internode_communicator

        if not hierarchical or (inter_comm is not None):
            self.backend_tensor.copy_(peer_weight)

        if hierarchical:
            broadcast(self.backend_tensor, 0, intra_comm)

    def append_low_precision_decentralized_synchronous_op(
        self,
        weight: BaguaTensor,
        left_peer_weight: BaguaTensor,
        right_peer_weight: BaguaTensor,
        hierarchical: bool = True,
        compression: str = "MinMaxUInt8",
    ) -> BaguaBucket:
        """
        Append a low precision decentralized synchronous operation to a bucket. It will compress the difference
        of local models between two successive iterations and exchange them among workers.

        The operations will be executed by the Bagua backend in the order they are appended
        when all the tensors within the bucket are marked ready.

        Args:
            weight (BaguaTensor): Model replica of current worker's local model. It should be of the same size
                with the bucket tensors total size. Use ``self.flattened_tensor().to_bagua_tensor(...)`` to create such a tensor.
            left_peer_weight (BaguaTensor): Model replica of current worker's left peer. It should be of the same size
                with the bucket tensors total size. Use ``self.flattened_tensor().to_bagua_tensor(...)`` to create such a tensor,
                then copy the initializing weights of current worker's left peer to the tensor.
            right_peer_weight (BaguaTensor): Model replica of current worker's right peer. It should be of the same size
                with the bucket tensors total size. Use ``self.flattened_tensor().to_bagua_tensor(...)`` to create such a tensor.
                then copy the initializing weights of current worker's right peer to the tensor.
            hierarchical (bool): Enable hierarchical communication. Which means the GPUs on the same machine
                will communicate will each other first. After that, machines do inter-node communication. This can
                boost performance when the inter-node communication cost is high.
            compression (str): The way how tensors are compressed for communication. Currently "MinMaxUInt8" is supported.
        Returns:
            The bucket itself.
        """

        if hierarchical:
            self.backend_bucket.append_low_precision_decentralized_synchronous_op(
                self._bagua_backend.internode_communicator,
                self._bagua_backend.intranode_communicator,
                hierarchical=hierarchical,
                peer_selection_mode="ring",
                compression=compression,
                weight=weight._bagua_backend_tensor,
                left_peer_weight=left_peer_weight._bagua_backend_tensor,
                right_peer_weight=right_peer_weight._bagua_backend_tensor,
            )
        else:
            self.backend_bucket.append_low_precision_decentralized_synchronous_op(
                self._bagua_backend.global_communicator,
                None,
                hierarchical=hierarchical,
                peer_selection_mode="ring",
                compression=compression,
                weight=weight._bagua_backend_tensor,
                left_peer_weight=left_peer_weight._bagua_backend_tensor,
                right_peer_weight=right_peer_weight._bagua_backend_tensor,
            )

        return self

    def append_asynchronous_model_average_op(self, peer_selection_mode: str):
        """
        Append an asynchronous model average operation to a bucket. This operation will enable continuous model averaging between workers
        while training a model.
        The operations will be executed by the Bagua backend in the order they are appended
        when all the tensors within the bucket are marked ready.

        Args:
            peer_selection_mode (str): The way how workers communicate with each otehr. Currently "all" is supported.
                "all" means all workers' weights are averaged during each communication.
        """

        self.backend_bucket.append_decentralized_asynchronous_op(
            self._bagua_backend.global_communicator,
            None,
            peer_selection_mode=peer_selection_mode,
            torch_stream=torch.cuda.current_stream().cuda_stream,
        )
        return self

    def clear_ops(self) -> BaguaBucket:
        """
        Clear the previously appended operations.
        """
        self.backend_bucket.clear_ops()
        return self

    def bytes(self) -> int:
        """Returns the total number of bytes occupied by the bucket.

        Returns:
            int: number of bucket bytes
        """
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)
