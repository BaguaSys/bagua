from torch.utils.data.dataset import Dataset
from .cache_loader import CacheLoader


class CacheDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        backend: str = "redis",
        capacity_per_node: int = 100_000_000_000,
        key_prefix: str = "",
        batch_reads: int = 1,
        batch_writes: int = 20,
        **kwargs,
    ):
        """ """

        self.dataset = dataset

        self.cache_loader = CacheLoader(
            backend,
            capacity_per_node,
            key_prefix,
            batch_reads,
            batch_writes,
            **kwargs,
        )

    def __getitem__(self, item):
        return self.cache_loader.get(item, lambda x: self.dataset[x])

    def __len__(self):
        return len(self.dataset)
