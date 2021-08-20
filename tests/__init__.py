import torch
import unittest


def skip_if_cuda_available():
    if torch.cuda.is_available():
        return unittest.skip("skip when cuda is available")

    return lambda func: func
