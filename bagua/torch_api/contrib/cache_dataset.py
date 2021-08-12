from torch.utils.data.dataset import Dataset
from .cache_loader import CacheLoader

__all__ = ["CacheDataset"]


class CacheDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        backend: str = "redis",
        key_prefix: str = "",
        batch_writes: int = 20,
        **kwargs,
    ):
        """
        A dataset wrapper which caches `dataset` samples.

        This is useful in scenarios when `dataset` has a lot preprocessing work to fetch a sample.

        Args:
            dataset: Dataset used for caching.
            backend(str): The backend to use. Currently "redis" is supported.
            key_prefix(str): Prefix of the cache key. Default ``""``.
            batch_writes(int): How many key-value pairs written to cache once. Default ``20``.

        Example::

            >>> from bagua.torch_api.contrib import CacheDataset
            >>> cache_dataset = CacheDataset(
            ...     dataset, backend="redis", hosts=None, cluster_mode=False
            ... )
            >>> dataloader = torch.utils.data.DataLoader(cached_dataset)

        .. note::

            This class use :class:`CacheLoader` as the implementation of cache. See :class:`CacheLoader` for more information.

        .. note::
            The cache assocaite dataset indices to determined dataset samples, thus it will violate the randomness of the dataset.
            Use :class:`CacheLoader` which can wrap arbitrary data loading logic in this situation.

        """

        self.dataset = dataset

        self.cache_loader = CacheLoader(backend, key_prefix, batch_writes, **kwargs,)

    def __getitem__(self, item):
        return self.cache_loader.get(item, lambda x: self.dataset[x])

    def __len__(self):
        return len(self.dataset)
