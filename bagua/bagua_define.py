import enum
from typing import List
import sys
if sys.version_info >= (3, 9):
    from typing import TypedDict  # pytype: disable=not-supported-yet
else:
    from typing_extensions import TypedDict
from pydantic import BaseModel


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

    def update(self, param_dict: dict):
        tmp = self.dict()
        tmp.update(param_dict)
        self.parse_obj(tmp)
        return self
