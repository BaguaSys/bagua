import pickle
from collections import defaultdict

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
        batch_writes: int = 1,
        **kwargs,
    ):
        """
        A mapping from keys to values. Values are automatically loaded by the cache, and
        are stored in the cache until evicted.

        Args:
            backend(str): The backend to use. Currently "redis" is supported, which means to use :class:`RedisStore`.
            key_prefix(str): Prefix of the cache key. Default ``""``.
            batch_writes(int): How many key-value pairs written to cache once. Default ``1``.

        Example::
            >>> # redis server '127.0.0.1:7000' must be alive beforehand
            >>> hosts = [{"host": "127.0.0.1", "port": "7000"}]
            >>> loader = CacheLoader(backend="redis", hosts=hosts, cluster_mode=False)
            >>>
            >>> loader.get(index, lambda x: items[x])
        """

        self.backend = backend
        self.key_prefix = key_prefix

        if backend == "redis":
            from .utils.redis_store import RedisStore

            self.store = RedisStore(**kwargs)
        else:
            raise ValueError('invalid backend, only support "redis" currently')

        self.fetcher = BatchFetcher(self.store, 1, batch_writes)

    def get(self, key, load_fn):
        """
        Returns the value associated with key in cache, first loading the value by calling `load_fn(key)` if necessary.
        """

        cache_key = "{}{}".format(self.key_prefix, key).encode()
        ret = self.fetcher.read(cache_key)

        if ret is None:
            ret = load_fn(key)
            # write to store
            self.fetcher.write(cache_key, ret)
        return ret

    def num_keys(self):
        """Returns total number of keys in cache"""

        return self.store.num_keys()

    def cleanup(self):
        """Cleanup the resources used."""

        # TODO: cleanup automatically
        self.store.shutdown()


class BatchFetcher:
    def __init__(self, store, batch_reads, batch_writes):
        self.store = store
        self.batch_reads = max(1, batch_reads)
        self.batch_writes = max(1, batch_writes)

        self.write_map = defaultdict()
        self.write_cnt = 0
        self.read_cnt = 0

        self.last_write_tms = None

    def read(self, key):
        self.read_cnt += 1

        try:
            ret = self.store.get(key)
        except:
            ret = None
        else:
            self.write_post_read()

        if ret is not None:
            return deserialize(ret)
        return ret

    def write(self, key, value):
        self.write_cnt += 1

        self.write_map[key] = serialize(value)
        if self.write_cnt % self.batch_writes == 0:
            self.flush_write_map()

    def write_post_read(self):
        if self.read_cnt % 1000 == 0 and len(self.write_map) > 0:
            self.flush_write_map()

    def flush_write_map(self):
        try:
            self.store.mset(self.write_map)
        except:
            pass
        else:
            self.write_map.clear()
