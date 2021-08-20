import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
from bagua.torch_api.contrib import (
    LoadBalancingDistributedSampler,
    LoadBalancingDistributedBatchSampler,
)
from tests import skip_if_cuda_available


class TestLoadBalancingDataLoader(unittest.TestCase):
    @skip_if_cuda_available()
    def test_load_balancing_distributed_sampler(self):
        n = 10
        dataset = TensorDataset(torch.randn(n, 2), torch.randperm(n))
        sampler = LoadBalancingDistributedSampler(
            dataset, complexity_fn=lambda x: x[1], num_replicas=1, rank=0, shuffle=False
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=sampler is None,
            sampler=sampler,
        )

        for i, data in enumerate(dataloader):
            self.assertTrue(i == data[1].item())

    @skip_if_cuda_available()
    def test_load_balancing_distributed_batch_sampler(self):
        num_replicas = 1
        total_batch = 5

        n = sum([i + 1 for i in range(total_batch)]) * num_replicas
        dataset = TensorDataset(torch.randn(n, 2), torch.randperm(n))

        def batch_fn(indices):
            batch = []
            batch_size = 1
            for i in indices:
                batch.append(i)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
                    batch_size += 1
            if len(batch) > 0:
                yield batch

        sampler = LoadBalancingDistributedSampler(
            dataset,
            complexity_fn=lambda x: x[1],
            num_replicas=num_replicas,
            rank=0,
            shuffle=False,
        )

        batch_sampler = LoadBalancingDistributedBatchSampler(
            sampler,
            batch_fn=lambda x: list(batch_fn(x)),
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)

        cur_idx = 0
        for i, data in enumerate(dataloader):
            batch_size = data[0].shape[0]
            self.assertTrue(batch_size == i + 1)
            self.assertTrue(cur_idx == data[1][0].item())

            cur_idx += batch_size * num_replicas


if __name__ == "__main__":
    unittest.main()
