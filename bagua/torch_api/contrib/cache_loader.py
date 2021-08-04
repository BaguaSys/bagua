import pickle
from collections import defaultdict


def serialize(input):
    return pickle.dumps(input)


def deserialize(input):
    return pickle.loads(input)


class CacheLoader:
    def __init__(
        self,
        backend: str = "redis",
        capacity_per_node: int = 100_000_000_000,
        key_prefix: str = "",
        batch_reads: int = 1,
        batch_writes: int = 20,
        **kwargs,
    ):
        """ """
        self.backend = backend
        self.capacity_per_node = capacity_per_node
        self.key_prefix = key_prefix

        if backend == "redis":
            from .utils.redis_store import RedisStore

            self.store = RedisStore(capacity_per_node=capacity_per_node, **kwargs)
        else:
            raise ValueError(
                'invalid backend, only support "redis" and "lmdb" at present'
            )

        self.fetcher = BatchFetcher(self.store, batch_reads, batch_writes)

    def get(self, key, load_fn):
        cache_key = "{}{}".format(self.key_prefix, key).encode()
        ret = self.fetcher.read(cache_key)

        if ret == None:
            ret = load_fn(key)
            # write to store
            self.fetcher.write(cache_key, ret)
        return ret

    def num_keys(self):
        return self.store.num_keys()

    def cleanup(self):
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

        ret = self.store.get(key)
        self.write_post_read()
        if ret is not None:
            return deserialize(ret)
        return ret

    def write(self, key, value):
        self.write_cnt += 1

        self.write_map[key] = serialize(value)
        if self.write_cnt % self.batch_writes == 0:
            self.store.mset(self.write_map)
            self.write_map.clear()

    def write_post_read(self):
        if self.read_cnt % 1000 == 0 and len(self.write_map) > 0:
            self.store.mset(self.write_map)
            self.write_map.clear()
