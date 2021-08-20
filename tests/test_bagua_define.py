import unittest
from bagua.bagua_define import (
    TensorDeclaration,
    BaguaHyperparameter,
    TensorDtype,
)
from tests import skip_if_cuda_available


class TestBaguaDefine(unittest.TestCase):
    @skip_if_cuda_available()
    def test_bagua_hyperparameter(self):
        hp = BaguaHyperparameter(is_hierarchical_reduce=False)

        buckets = [
            [
                TensorDeclaration(
                    {"name": "A", "num_elements": 123, "dtype": TensorDtype.F32}
                )
            ]
        ]

        hp = hp.update(
            {
                "buckets": buckets,
            }
        )

        assert hp.buckets == buckets
        assert hp.is_hierarchical_reduce is False


if __name__ == "__main__":
    unittest.main()
