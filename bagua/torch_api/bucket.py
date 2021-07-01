#!/usr/bin/env python3

from typing import List

import bagua_core as B
import torch
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

        if alignment > 1:
            padding = sum(tensor.numel() for tensor in self.tensors) % alignment
            if padding > 0:
                padding = alignment - padding
                pad_tensor = torch.zeros(
                    padding, dtype=self.tensors[0].dtype, device=self.tensors[0].device
                )
                self.tensors.append(
                    pad_tensor.to_bagua_tensor("padding_tensor_bucket_" + name)
                )

        self.backend_tensor = None
        self.flatten = flatten
        if self.flatten:
            self._flatten_()
        self.name = name

        self.backend_bucket = B.BaguaBucketPy(
            name,
            [tensor._bagua_backend_tensor for tensor in tensors],
        )

        for tensor in self.tensors:
            tensor._bagua_bucket = self

    def _flatten_(self):
        """
        Flatten inner tensors in place.
        """
        if self.check_flatten():
            return

        if len(self.tensors) == 0:
            return
        total_size = 0
        for tensor in self.tensors:
            total_size += tensor.numel()

        flatten_tensor = torch.zeros(total_size, dtype=self.tensors[0].dtype).to(
            self.tensors[0].device
        )
        flatten_storage = flatten_tensor.storage()

        offset = 0
        for tensor in self.tensors:
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
        return check_contiguous(self.tensors)
