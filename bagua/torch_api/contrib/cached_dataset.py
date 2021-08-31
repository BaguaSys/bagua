from torch.utils.data.dataset import Dataset
from .cache_loader import CacheLoader

__all__ = ["CachedDataset"]


class CachedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        backend: str = "redis",
        dataset_name: str = "",
        writer_buffer_size: int = 20,
        **kwargs,
    ):
        """
        Cached dataset wraps a `PyTorch dataset <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_
        to cache its samples in memory, so that accessing these samples after the
        first time can be much faster. This is useful when samples need tedious preprocessing to produce, or reading
        the dataset itself is slow, which could slow down the whole training process.

        Internally, the samples are indexed by a string key ``"{dataset_name}_{index}"`` and saved in a distributed key-value
        store, where ``dataset_name`` is specified when initializing the cached dataset, and ``index`` is the index
        of a specific sample (the argument of :meth:`__getitem__` method in a PyTorch dataset).

        Args:
            dataset: PyTorch dataset to be wrapped.
            backend(str): Backend distributed key-value store implementation. Can be ``"redis"``.
            dataset_name(str): Name of the dataset. Default ``""``.
            writer_buffer_size(int): Number of samples to collect before writing to the backend key-value store.
                Useful for improving the backend throughput.

        Example::

            >>> from bagua.torch_api.contrib import CachedDataset
            >>> cache_dataset = CachedDataset(dataset, backend="redis", dataset_name="ds")
            >>> dataloader = torch.utils.data.DataLoader(cached_dataset)

        .. note::
            Cached dataset is a special case of cache loader. Parameter :attr:`backend` and :attr:`writer_buffer_size` in
            initializing a cached dataset have the same meanings as those in initializing a cache loader. You can
            provide the arguments for cache loader here in ``**kwargs``. See also :class:`~bagua.torch_api.contrib.cache_loader.CacheLoader`.

        """

        self.dataset = dataset

        self.cache_loader = CacheLoader(
            backend,
            dataset_name,
            writer_buffer_size,
            **kwargs,
        )
        """
        The backend cache loader instance.
        """

    def __getitem__(self, item):
        return self.cache_loader.get(item, lambda x: self.dataset[x])

    def __len__(self):
        return len(self.dataset)
