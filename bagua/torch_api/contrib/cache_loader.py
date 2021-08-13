import pickle
from collections import defaultdict
import atexit

__all__ = ["CacheLoader"]


def serialize(input):
    return pickle.dumps(input)


def deserialize(input):
    return pickle.loads(input)


class CacheLoader:
    def __init__(
        self,
        backend: str = "redis",
        key_prefix: str = "",
        writer_buffer_size: int = 1,
        **kwargs,
    ):
        """
        A mapping from keys to values. Values are automatically loaded by the cache, and
        are stored in the cache until evicted.

        Internally, values are indexed by ``"{key_prefix}_{key}"`` and saved in a distributed Key-Value
        store, where ``key_prefix`` is specified on initializing, and ``key`` is the argument in :func:`get`.

        By default, CacheLoader uses :class:`RedisStore` as its backend distributed Key-Value store implementation. It
        could reuse a list of initialized redis servers or bootstrap local redis servers by itself. See
        :class:`bagua.torch_api.contrib.utils.redis_store.RedisStore` for further customization.

        Args:
            backend(str): Backend distributed Key-Value store implementation. Can be ``"redis"``.
            key_prefix(str): Prefix added to the cache key. Better to be short. Default ``""``.
            writer_buffer_size(int): Number of samples to collect before writing to the backend Key-Value store.
                Useful for improving the backend throughput.

        Example::
            To reuse a list of initialized redis servers for "redis" backend:

            >>> from bagua.torch_api.contrib import CacheLoader
            >>>
            >>> hosts = [{"host": "192.168.1.0", "port": "7000"}, {"host": "192.168.1.1", "port": "7000"}]
            >>> loader = CacheLoader(backend="redis", hosts=hosts, cluster_mode=True, key_prefix="test")
            >>>
            >>> loader.get(index, lambda x: items[x])

            To bootstrap local redis servers for "redis" backend:

            >>> loader = CacheLoader(backend="redis", hosts=None, cluster_mode=True, capacity_per_node=100000000)

        .. note::
            Setting a specific `key_prefix` can be useful to avoid overwriting existing cache data.

        """

        self.backend = backend
        self.key_prefix = key_prefix

        if backend == "redis":
            from .utils.redis_store import RedisStore

            self.store = RedisStore(**kwargs)
        else:
            raise ValueError('invalid backend, only support "redis" currently')

        self.fetcher = BatchFetcher(self.store, 1, writer_buffer_size)
        self.register_shutdown_handler()

    def get(self, key, load_fn):
        """
        Returns the value associated with key in cache, first loading the value if necessary.
        `load_fn` accepts `key` as input, and returns the data to be serialized and stored.
        """

        cache_key = "{}{}".format(self.key_prefix, key)
        ret = self.fetcher.read(cache_key)

        if ret is None:
            ret = load_fn(key)
            # write to store
            self.fetcher.write(cache_key, ret)
        return ret

    def num_keys(self):
        """Returns the total number of keys in cache"""

        return self.store.num_keys()

    def register_shutdown_handler(self):
        atexit.register(self.store.shutdown)


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
