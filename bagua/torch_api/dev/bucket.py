#!/usr/bin/env python3

from typing import List

import bagua_core as B
import torch
from bagua.torch_api.dev.tensor import BaguaTensor
from bagua.torch_api.utils import check_contiguous


class BaguaBucket:
    def __init__(self, tensors: List[BaguaTensor], flatten: bool, bucket_index: int) -> None:
        self.tensors = tensors
        self.backend_tensor = None
        self.flatten = flatten
        if flatten:
            self.flatten_()

        self.backend_bucket = B.BaguaBucketPy(
            "bucket_" + str(bucket_index),
            [tensor.backend_tensor for tensor in tensors],
            inplace=True,
            align_bytes=1,
        )

    def flatten_(self):
        """
        flatten inner tensors in place
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
        return check_contiguous(self.tensors)
