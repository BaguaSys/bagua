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
        CachedDataset wraps a PyTorch Dataset to cache its samples in memory, so that accessing these samples after the
        first time can be much faster. This is useful when samples need tedious preprocessing to produce, or reading
        the dataset itself is slow, which could slow down the whole training process.

        Internally, the samples are indexed by a key ``"{dataset_name}_{index}"`` and saved in a distributed Key-Value
        store, where ``dataset_name`` is specified when initializing the CachedDataset, and ``index`` is the index
        of a specific sample (the argument of `__getitem__(...)` method in a PyTorch Dataset).

        Args:
            dataset: PyTorch Dataset to be wrapped.
            backend(str): Backend distributed Key-Value store implementation. Can be ``"redis"``.
            dataset_name(str): Name of the dataset. Better to be short. Default ``""``.
            writer_buffer_size(int): Number of samples to collect before writing to the backend Key-Value store.
                Useful for improving the backend throughput.

        Example::

            >>> from bagua.torch_api.contrib import CachedDataset
            >>> cache_dataset = CachedDataset(dataset, backend="redis", dataset_name="ds")
            >>> dataloader = torch.utils.data.DataLoader(cached_dataset)

        .. note::
            `CachedDataset` is a special case of `CacheLoader`, and parameter `backend` and `writer_buffer_size`
            in `CachedDataset` have the same meanings with those in `CacheLoader`. Further customization can be found in
            :class:`bagua.torch_api.contrib.CacheLoader`.

        """

        self.dataset = dataset

        self.cache_loader = CacheLoader(
            backend, dataset_name, writer_buffer_size, **kwargs,
        )
        """
        The backend cache instance.
        """

    def __getitem__(self, item):
        return self.cache_loader.get(item, lambda x: self.dataset[x])

    def __len__(self):
        return len(self.dataset)
