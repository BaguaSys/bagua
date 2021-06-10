from enum import Enum


class BucketType(Enum):
    """
    An enum-like class of type for buckets used in communication-computation
    overlapping process.
    """

    Gradient = 0
    Weight = 1
    Param = 2


class ReduceOp(Enum):
    Sum = 0
    Average = 1
