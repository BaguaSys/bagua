import torch
from torch.utils.data import TensorDataset
from bagua.torch_api.tools.data import (
    LoadBalancingDistributedSampler,
    LoadBalancingDistributedBatchSampler,
)
import logging


logging.basicConfig(level=logging.DEBUG)


def test_loadbalance():
    n = 100
    dataset = TensorDataset(torch.randn(n, 2), torch.arange(n))
    sampler = LoadBalancingDistributedSampler(
        dataset, lambda x: 2 ** (x[1].item() - 1).bit_length(), num_replicas=2, rank=0
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        sampler=sampler,
    )

    for i, data in enumerate(dataloader):
        print(i, data)


def test_batchsampler():
    n = 20
    dataset = TensorDataset(torch.randn(n, 2), torch.arange(n))

    import random

    def batch_fn(indices):
        batch = []
        batch_size = random.randint(1, 4)
        for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                yield batch
                batch = []
                batch_size = random.randint(1, 4)
        if len(batch) > 0:
            yield batch

    sampler = LoadBalancingDistributedSampler(
        dataset,
        lambda x: 2 ** (x[1].item() - 1).bit_length(),
        num_replicas=2,
        rank=0,
    )

    batch_sampler = LoadBalancingDistributedBatchSampler(
        sampler,
        batch_fn=lambda x: list(batch_fn(x)),
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)

    for epoch in range(10):
        print("Training ....")
        batch_sampler.set_epoch(epoch)

        for i, data in enumerate(dataloader):
            print(i, data)


if __name__ == "__main__":
    test_loadbalance()
    test_batchsampler()
