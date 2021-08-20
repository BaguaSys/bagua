import torch
import unittest
from functools import wraps


def cpuTest(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            return unittest.skip(
                "skip cpu test: {}.{}".format(func.__module__, func.__name__)
            )

        return func(*args, **kwargs)

    return wrapper
