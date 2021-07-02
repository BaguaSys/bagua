#!/usr/bin/env python3

from __future__ import annotations
from typing import List, Callable, Optional

import bagua_core as B
import torch
from bagua.torch_api.globals import _get_global_state

from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.utils import check_contiguous


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
            flatten: If True, flatten the input tensors so that they are
                contiguous in memory.
            alignment: If alignment > 1, Bagua will create a padding tensor to
                the bucket so that the total number of elements in the bucket divides
                the given alignment.
        """
        self.tensors = tensors
        """
        The tensors contained within the bucket.
        """
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
                with torch.cuda.stream(_get_global_state().get_communication_stream()):
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
            average (bool): If True, the gradients on each worker are averaged. Otherwise, they are summed.
            scattergather (bool): If true, the communication between workers are done with scatter gather instead
                of allreduce. This is required for using compression.
            compression: If not None, the tensors will be compressed for communication. Currently "MinMaxUInt8" is
                supported.
        Returns:
            The bucket itself.
        """
        if hierarchical:
            self.backend_bucket.append_centralized_synchronous_op(
                _get_global_state().get_internode_communicator(),
                _get_global_state().get_intranode_communicator(),
                hierarchical=hierarchical,
                average=average,
                scattergather=scattergather,
                compression=compression,
            )
        else:
            self.backend_bucket.append_centralized_synchronous_op(
                _get_global_state().get_global_communicator(),
                None,
                hierarchical=hierarchical,
                average=average,
                scattergather=scattergather,
                compression=compression,
            )
        return self

    def append_decentralized_synchronous_op(
        self,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
        communication_interval: int = 1,
    ) -> BaguaBucket:
        """
        Append a decentralized synchronous operation to a bucket. It will do gossipy style model averaging among workers.

        The operations will be executed by the Bagua backend in the order they are appended
        when all the tensors within the bucket are marked ready.

        Args:
            hierarchical (bool): Enable hierarchical communication. Which means the GPUs on the same machine
                will communicate will each other first. After that, machines do inter-node communication. This can
                boost performance when the inter-node communication cost is high.
            peer_selection_mode (str): Can be "all" or "shift_one". "all" means all workers'
                weights are averaged in each communication step. "shift_one" means each worker
                selects a different peer to do weights average in each communication step.
            communication_interval (int): Number of iterations between two communication steps.
        Returns:
            The bucket itself.
        """
        self.backend_bucket.append_decentralized_synchronous_op(
            _get_global_state().get_internode_communicator(),
            _get_global_state().get_intranode_communicator(),
            hierarchical=hierarchical,
            compression=None,
            peer_selection_mode=peer_selection_mode,
            communication_interval=communication_interval,
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
