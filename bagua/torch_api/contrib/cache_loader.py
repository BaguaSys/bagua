import pickle
from collections import defaultdict
from typing import Callable


__all__ = ["CacheLoader"]


def serialize(input):
    return pickle.dumps(input)


def deserialize(input):
    return pickle.loads(input)


class CacheLoader:
    def __init__(
        self,
        backend: str = "redis",
        dataset_name: str = "",
        writer_buffer_size: int = 1,
        **kwargs,
    ):
        """
        Cache loader caches values calculated by an expensive function by theirs keys via :meth:`get`,
        so that the values can be retrieved faster next time.

        Internally, values are indexed by ``"{dataset_name}_{key}"`` and saved in a distributed key-value
        store, where ``dataset_name`` is specified on initializing, and ``key`` is the argument in :meth:`get`.

        By default, cache loader uses :class:`~bagua.torch_api.contrib.utils.redis_store.RedisStore` as its backend distributed key-value store implementation. It
        supports using a list of existing redis servers or spawning new redis servers. Parameters for :class:`~bagua.torch_api.contrib.utils.redis_store.RedisStore` can be provided here in
        ``**kwargs``.

        Args:
            backend(str): Backend distributed key-value store implementation. Can be ``"redis"``.
            dataset_name(str): Name of the dataset. Default ``""``.
            writer_buffer_size(int): Number of samples to collect before writing to the backend key-value store.
                Useful for improving the backend throughput.

        Example::
            To use a list of existing redis servers for the "redis" backend:

            >>> from bagua.torch_api.contrib import CacheLoader
            >>>
            >>> hosts = [{"host": "192.168.1.0", "port": "7000"}, {"host": "192.168.1.1", "port": "7000"}]
            >>> loader = CacheLoader(backend="redis", hosts=hosts, cluster_mode=True, dataset_name="test")
            >>>
            >>> loader.get(index, lambda x: items[x])

            To spawn new redis servers for the "redis" backend:

            >>> loader = CacheLoader(backend="redis", hosts=None, cluster_mode=True, capacity_per_node=100000000)

        .. note::
            Cache loaders with the same :attr:`dataset_name` will reuse and overwrite each other's cache.
            Use a different :attr:`dataset_name` if this is not desired.

        """

        self.backend = backend
        self.dataset_name = dataset_name

        if backend == "redis":
            from .utils.redis_store import RedisStore

            self.store = RedisStore(**kwargs)
        else:
            raise ValueError('Invalid backend, only support "redis" currently')

        self.fetcher = BatchFetcher(self.store, 1, writer_buffer_size)

    def get(self, key: str, load_fn: Callable[[str], None]):
        """
        Returns the value associated with :attr:`key` in cache, use  :attr:`load_fn` to create the entry if the key does not exist
        in the cache. :attr:`load_fn` is a function taking :attr:`key` as its argument, and returning corresponding value to
        be cached.
        """

        cache_key = "{}_{}".format(self.dataset_name, key)
        ret = self.fetcher.read(cache_key)

        if ret is None:
            ret = load_fn(key)
            # write to store
            self.fetcher.write(cache_key, ret)
        return ret

    def num_keys(self):
        """Returns the number of keys in the cache."""

        return self.store.num_keys()


class BatchFetcher:
    def __init__(self, store, read_buffer_size, writer_buffer_size):
        self.store = store
        self.read_buffer_size = max(1, read_buffer_size)
        self.writer_buffer_size = max(1, writer_buffer_size)

        self.write_map = defaultdict()
        self.write_cnt = 0
        self.read_cnt = 0

        self.last_write_tms = None

    def read(self, key):
        self.read_cnt += 1

        try:
            ret = self.store.get(key)
        except Exception:
            ret = None
        else:
            self.write_post_read()

        if ret is not None:
            return deserialize(ret)
        return ret

    def write(self, key, value):
        self.write_cnt += 1

        self.write_map[key] = serialize(value)
        if self.write_cnt % self.writer_buffer_size == 0:
            self.flush_write_map()

    def write_post_read(self):
        if self.read_cnt % 1000 == 0 and len(self.write_map) > 0:
            self.flush_write_map()

    def flush_write_map(self):
        try:
            self.store.mset(self.write_map)
        except Exception:
            pass
        else:
            self.write_map.clear()
