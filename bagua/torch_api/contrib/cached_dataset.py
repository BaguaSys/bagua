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
        overwrite=True,
        capacity_per_node: int = 10_000_000_000,
        **kwargs,
    ):
        self.dataset = dataset
        self.backend = backend

        if backend == "redis":
            from .utils.redis_store import RedisStore

            self.store = RedisStore(
                overwrite=overwrite, capacity_per_node=capacity_per_node, **kwargs
            )
        elif backend == "lmdb":
            from .utils.lmdb_store import LmdbStore

            self.store = LmdbStore(
                overwrite=overwrite, capacity_per_node=capacity_per_node, **kwargs
            )
        else:
            raise ValueError(
                'invalid backend, only support "redis" and "lmdb" at present'
            )

    def __getitem__(self, item):
        ret = self.store.get(str(item).encode())

        if ret is not None:
            return deserialize(ret)

        # write to store
        ret = self.dataset[item]
        self.store.set(str(item).encode(), serialize(ret))
        return ret

    def __len__(self):
        return len(self.dataset)

    def cleanup(self):
        self.store.shutdown()
