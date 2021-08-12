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

        Current backend is "redis". Using "redis" backend, the cache will initialize an instance of :class:`RedisStore`
        by a list of initialized redis servers or bootstrap redis servers locally. See :class:`bagua.torch_api.contrib.utils.redis_store.RedisStore`
        for more information.

        Args:
            backend(str): The backend to use. Currently ``"redis"`` is supported.
            key_prefix(str): Prefix of the cache key. Default ``""``.
            batch_writes(int): How many key-value pairs written to cache once. Default ``1``. If `batch_writes > 1`, the
                cache will delay writing non-existed key-value pairs until `batch_writes` key-value pairs are accumulated.
                Thus it could combine multiple `set` operations to one `mset` operation. This is expected to reduce
                the write latency.

        Example::
            To use "redis" backend and initialized redis clusters: `{'192.168.1.0:7000', '192.168.1.1:7000'}`:

            >>> hosts = [{"host": "192.168.1.0", "port": "7000"}, {"host": "192.168.1.1", "port": "7000"}]
            >>> loader = CacheLoader(backend="redis", hosts=hosts, cluster_mode=True)
            >>>
            >>> loader.get(index, lambda x: items[x])

            To use "redis" backend and bootstrap redis servers locally:

            >>> loader = CacheLoader(backend="redis", hosts=None, cluster_mode=True, capacity_per_node=100000000)
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
        Returns the value associated with key in cache, first loading the value if necessary.
        `load_fn` accepts `key` as input, and returns an object ser
        """

        cache_key = "{}{}".format(self.key_prefix, key).encode()
        ret = self.fetcher.read(cache_key)

        if ret is None:
            ret = load_fn(key)
            # write to store
            self.fetcher.write(cache_key, ret)
        return ret

    def num_keys(self):
        """Returns the total number of keys in cache"""

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
        if self.write_cnt % self.batch_writes == 0:
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
