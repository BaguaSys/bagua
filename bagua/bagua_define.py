import enum
import json
from typing import List
import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict
from pydantic import BaseModel


class DistributedAlgorithm(enum.Enum):
    """
    An enum-like class of available distributed algorithms: allreduce, sg-allreduce, quantize and
    decentralize.

    The values of this class are lowercase strings, e.g., ``"allreduce"``. They can
    be accessed as attributes, e.g., ``DistributedAlgorithm.GradientAllReduce``.

    This class can be directly called to parse the string, e.g.,
    ``DistributedAlgorithm(algor_str)``.
    """

    GradientAllReduce = "allreduce"
    ScatterGatherAllReduce = "sg-allreduce"
    Decentralize = "decentralize"
    QuantizeAllReduce = "quantize"

    @staticmethod
    def from_str(val: str):
        if not isinstance(val, str):
            raise ValueError(
                "DistributedAlgorithm name must be a string, but got: {}".format(val)
            )

        reverse_dict = {e.value: e for e in DistributedAlgorithm}
        return reverse_dict[val]


class TensorDtype(str, enum.Enum):
    F32 = "f32"
    F16 = "f16"
    U8 = "u8"


class TensorDeclaration(TypedDict):
    name: str
    num_elements: int
    dtype: TensorDtype


class BaguaHyperparameter(BaseModel):
    """
    Structured all bagua hyperparameters
    """

    buckets: List[List[TensorDeclaration]] = []
    is_hierarchical_reduce: bool = False
    distributed_algorithm: str = DistributedAlgorithm.GradientAllReduce.value

    def update(self, param_dict: dict):
        tmp = self.dict()
        tmp.update(param_dict)
        self.parse_obj(tmp)

        return self
