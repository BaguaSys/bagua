from torch.utils.data.dataset import Dataset
from .cache_loader import CacheLoader

__all__ = ["CacheDataset"]


class CacheDataset(Dataset):
    """
    A dataset wrapper which caches `dataset` samples.

    Args:
        dataset: Dataset used for caching.
        backend(str): The backend to use. Currently "redis" is supported, which means to use :class:`RedisStore`.
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
    """

    def __init__(
        self,
        dataset: Dataset,
        backend: str = "redis",
        key_prefix: str = "",
        batch_writes: int = 20,
        **kwargs,
    ):

        self.dataset = dataset

        self.cache_loader = CacheLoader(backend, key_prefix, batch_writes, **kwargs,)

    def __getitem__(self, item):
        return self.cache_loader.get(item, lambda x: self.dataset[x])

    def __len__(self):
        return len(self.dataset)
