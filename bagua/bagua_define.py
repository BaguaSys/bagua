import enum
from typing import List
import sys

if sys.version_info >= (3, 9):
    from typing import TypedDict  # pytype: disable=not-supported-yet
else:
    from typing_extensions import TypedDict  # pytype: disable=not-supported-yet
from pydantic import BaseModel


class TensorDtype(str, enum.Enum):
    F32 = "f32"
    F16 = "f16"
    U8 = "u8"


class TensorDeclaration(TypedDict):
    name: str
    num_elements: int
    dtype: TensorDtype


def get_tensor_declaration_bytes(td: TensorDeclaration) -> int:
    dtype_unit_size = {
        TensorDtype.F32.value: 4,
        TensorDtype.F16.value: 2,
        TensorDtype.U8.value: 1,
    }

    return td["num_elements"] * dtype_unit_size[td["dtype"]]


class BaguaHyperparameter(BaseModel):
    """
    Structured all bagua hyperparameters
    """

    buckets: List[List[TensorDeclaration]] = []
    bucket_size: int = 0
    is_hierarchical_reduce: bool = False

    def update(self, param_dict: dict):
        tmp = self.dict()
        tmp.update(param_dict)
        for key, value in param_dict.items():
            if key in tmp:
                self.__dict__[key] = value

        return self
