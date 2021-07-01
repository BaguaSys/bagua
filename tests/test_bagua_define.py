import unittest
from bagua.bagua_define import (
    TensorDeclaration, BaguaHyperparameter, TensorDtype, DistributedAlgorithm)


class TestBaguaDefine(unittest.TestCase):
    def test_bagua_hyperparameter(self):
        hp = BaguaHyperparameter(
            is_hierarchical_reduce=False
        )

        buckets = [[TensorDeclaration(
            {"name": "A", "num_elements": 123, "dtype": TensorDtype.F32})]]

        hp = hp.update({
            "buckets": buckets,
            "distributed_algorithm": DistributedAlgorithm.Decentralize.value,
        })

        assert hp.buckets == buckets
        assert hp.is_hierarchical_reduce == False
        assert hp.distributed_algorithm == DistributedAlgorithm.Decentralize.value


if __name__ == "__main__":
    unittest.main()
