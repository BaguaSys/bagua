from __future__ import print_function

# BaguaTensor -> BaguaBucket -> set_comm_op -> BaguaCommBackend.register_buckets -> mark_comm_ready -> wait comm finished
#          BaguaCommunicator ->

import bagua_comm_core_py as B

B.show_version()
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


def test(chunk_size, num_elem, num_elem_allocated, target_chunk):
    p = torch.rand(num_elem_allocated, dtype=torch.float).cuda()
    bagua_tensor = B.BaguaTensor(
        ptr=p.data.data_ptr(),
        num_elem=num_elem,
        num_elem_allocated=p.data.numel(),
        dtype="f32",
        device_id=0,
    )
    before = bagua_tensor.to_numpy_f32().copy()
    before.resize(num_elem_allocated)
    before = before.reshape((chunk_size, -1))
    print(before)
    c = bagua_tensor.compress("min_max_uint8", chunk_size, target_chunk)
    com = c.to_numpy_u8().copy()
    com.resize(num_elem_allocated)
    com = com.reshape((chunk_size, -1))
    print(com)
    print(bagua_tensor.ptr(), c.ptr())

    p2 = torch.zeros(num_elem_allocated, dtype=torch.float).cuda()
    bagua_tensor2 = B.BaguaTensor(
        ptr=p2.data.data_ptr(),
        num_elem=num_elem,
        num_elem_allocated=p2.data.numel(),
        dtype="f32",
        device_id=0,
    )
    bagua_tensor2.decompress_from("min_max_uint8", chunk_size, c)
    after = bagua_tensor2.to_numpy_f32().copy()
    after.resize(num_elem_allocated)
    after = after.reshape((chunk_size, -1))
    print(after)
    print(abs(after - before) / abs(before))
    if target_chunk == -1:
        diff = abs(before - after).max() / abs(before).max()
    else:
        diff = (
            abs(before[target_chunk] - after[target_chunk]).max()
            / abs(before[target_chunk]).max()
        )
    print(diff)
    assert diff < 0.002


test(4, 95, 100, -1)
for i in range(4):
    test(4, 95, 100, i)
