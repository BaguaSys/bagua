from bagua.torch_api.contrib.cached_dataset import CachedDataset
from torch.utils.data.dataset import Dataset
import numpy as np
import logging
import unittest

logging.basicConfig(level=logging.DEBUG)


class TestDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.dataset = [(np.random.rand(5, 2), np.random.rand(1)) for _ in range(size)]

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return self.size


class TestCachedDataset(unittest.TestCase):
    def check_dataset(self, dataset, cached_dataset):
        for _, _ in enumerate(cached_dataset):
            pass

        self.assertEqual(cached_dataset.store.num_keys(), 10)
        for i in range(10):
            self.assertTrue((dataset[i][0] == cached_dataset[i][0]).all())
            self.assertTrue((dataset[i][1] == cached_dataset[i][1]).all())

    def test_lmdb(self):
        np.random.seed(0)
        dataset = TestDataset(10)
        cached_dataset = CachedDataset(
            dataset, backend="lmdb", path=".lmdb", overwrite=True
        )
        self.check_dataset(dataset, cached_dataset)

        cached_dataset.cleanup()

    def test_redis(self):
        np.random.seed(0)
        dataset = TestDataset(10)
        cached_dataset = CachedDataset(
            dataset, backend="redis", hosts=None, cluster_mode=False
        )
        self.check_dataset(dataset, cached_dataset)
        cached_dataset.cleanup()


if __name__ == "__main__":
    unittest.main()
