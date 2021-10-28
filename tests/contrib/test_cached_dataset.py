from bagua.torch_api.contrib.cached_dataset import CachedDataset
from torch.utils.data.dataset import Dataset
import numpy as np
import logging
import unittest
from tests import skip_if_cuda_available

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

        for i in range(len(dataset)):
            self.assertTrue((dataset[i][0] == cache_dataset[i][0]).all())
            self.assertTrue((dataset[i][1] == cache_dataset[i][1]).all())

    @skip_if_cuda_available()
    def test_redis(self):
        dataset1 = MyDataset(102)
        dataset2 = MyDataset(102)
        cache_dataset1 = CachedDataset(
            dataset1,
            backend="redis",
            dataset_name="d1",
        )
        cache_dataset2 = CachedDataset(
            dataset2,
            backend="redis",
            dataset_name="d2",
        )

        cache_dataset1.cache_loader.store.clear()

        self.check_dataset(dataset1, cache_dataset1)
        self.assertEqual(cache_dataset1.cache_loader.num_keys(), len(dataset1))

        self.check_dataset(dataset2, cache_dataset2)
        self.assertEqual(
            cache_dataset2.cache_loader.num_keys(), len(dataset1) + len(dataset2)
        )


if __name__ == "__main__":
    unittest.main()
