#!/usr/bin/env python3

import torch
import bagua_ext


def test_flatten():
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([4, 5, 6])
    t = bagua_ext.flatten([t1, t2])
    assert (t - torch.tensor([1, 2, 3, 4, 5, 6])).abs().sum() == 0


def test_unflatten():
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([4, 5, 6])
    t3, t4 = bagua_ext.unflatten(torch.tensor([3, 2, 1, 6, 5, 4]), [t1, t2])
    assert torch.allclose(t3, torch.tensor([3, 2, 1]))
    assert torch.allclose(t4, torch.tensor([6, 5, 4]))


if __name__ == "__main__":
    test_flatten()
    test_unflatten()
