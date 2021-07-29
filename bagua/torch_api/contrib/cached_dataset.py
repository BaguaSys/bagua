from torch.utils.data.dataset import Dataset
import pickle


def serialize(input):
    return pickle.dumps(input)


def deserialize(input):
    return pickle.loads(input)


class CachedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        backend: str = "redis",
        capacity_per_node: int = 10_000_000_000,
        key_prefix: str = "",
        **kwargs,
    ):
        """ """
        self.dataset = dataset
        self.backend = backend
        self.key_prefix = key_prefix

        if backend == "redis":
            from .utils.redis_store import RedisStore

            self.store = RedisStore(capacity_per_node=capacity_per_node, **kwargs)
        elif backend == "lmdb":
            from .utils.lmdb_store import LmdbStore

            self.store = LmdbStore(capacity_per_node=capacity_per_node, **kwargs)
        else:
            raise ValueError(
                'invalid backend, only support "redis" and "lmdb" at present'
            )

    def __getitem__(self, item):
        key = "{}{}".format(self.key_prefix, item).encode()

        ret = self.store.get(key)

        if ret is not None:
            return deserialize(ret)

        # write to store
        ret = self.dataset[item]
        self.store.set(key, serialize(ret))
        return ret

    def __len__(self):
        return len(self.dataset)

    def cleanup(self):
        self.store.shutdown()
