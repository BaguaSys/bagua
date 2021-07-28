from torch.utils.data.dataset import Dataset
import pyarrow as pa


def serialize(input):
    try:
        return pa.serialize(input).to_buffer()
    except Exception as e:
        raise RuntimeError("Serialization error!")

def deserialize(input):
    try:
        return pa.deserialize(input)
    except Exception as e:
        raise RuntimeError("Deserialization error!")


class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset, backend: str="redis", **kwargs):
        self.dataset = dataset
        self.backend = backend

        if backend == "redis":
            from .utils.redis_store import RedisStore
            self.store = RedisStore(**kwargs)
        elif backend == "lmdb":
            from .utils.lmdb_store import LmdbStore
            self.store = LmdbStore(**kwargs)
        else:
            raise ValueError("invalid backend, only support \"redis\" and \"lmdb\" at present")

    def __getitem__(self, item):
        value = self.store.get(str(item))

        if value is not None:
            return value

        # write to store
        value = self.dataset[item]
        self.store.set(str(item), serialize(value))
        return value

    def __len__(self):
        return len(self.dataset)
