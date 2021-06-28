#!/usr/bin/env python3

from typing import List

import bagua_core as B
import torch
from bagua.torch_api.dev.tensor import BaguaTensor
from bagua.torch_api.utils import check_contiguous


class BaguaBucket:
    def __init__(self, tensors: List[BaguaTensor], name: str, flatten: bool) -> None:
        """
        Create a Bagua bucket with a list of Bagua tensors.

        Args:
            tensors (List[BaguaTensor]): A list of Bagua tensors to be put in the
                bucket.
            name (str): The unique name of the bucket.
            flatten (bool): If True, flatten the input tensors so that they are
                contiguous in memory.
        """
        self.tensors = tensors
        self.backend_tensor = None
        self.flatten = flatten
        if flatten:
            self._flatten_()
        self.name = name

        self.backend_bucket = B.BaguaBucketPy(
            name,
            [tensor.backend_tensor for tensor in tensors],
            inplace=True,
            align_bytes=1,
        )

        for tensor in tensors:
            tensor._bagua_bucket = self

    def _flatten_(self):
        """
        Flatten inner tensors in place.
        """
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
        assert check_contiguous(self.tensors)

    def check_flatten(self) -> bool:
        """
        Returns True if the bucket's tensors are contiguous in memory.
        """
        return check_contiguous(self.tensors)
