from bagua.torch_api.contrib.cached_dataset import CachedDataset
from torch.utils.data.dataset import Dataset
import numpy as np
import logging
import unittest

logging.basicConfig(level=logging.DEBUG)


class MyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.dataset = [(np.random.rand(5, 2), np.random.rand(1)) for _ in range(size)]

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return self.size


class TestCacheDataset(unittest.TestCase):
    def check_dataset(self, dataset, cache_dataset):
        for _ in range(10):
            for _, _ in enumerate(cache_dataset):
                pass

        self.assertEqual(cache_dataset.cache_loader.num_keys(), len(dataset))
        for i in range(len(dataset)):
            self.assertTrue((dataset[i][0] == cache_dataset[i][0]).all())
            self.assertTrue((dataset[i][1] == cache_dataset[i][1]).all())

    def test_redis(self):
        dataset = MyDataset(102)
        cache_dataset = CachedDataset(
            dataset, backend="redis", hosts=None, cluster_mode=False
        )
        self.check_dataset(dataset, cache_dataset)


if __name__ == "__main__":
    unittest.main()
