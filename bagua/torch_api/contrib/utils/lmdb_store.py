import lmdb
from .store import Store
from typing import List, Dict, Optional


class LmdbStore(Store):
    def __init__(self, name, capacity_per_node: int = 1_000_000_000, overwrite=True):
        self.name = name
        self.capacity_per_node = capacity_per_node
        self.env = lmdb.open(self.name, map_size=self.capacity_per_node)

        if overwrite:
            self.clear()

    def set(self, key: str, value: str):
        with self.env.begin(write=True) as txn:
            txn.put(key, value)

    def get(self, key: str) -> Optional[str]:
        with self.env.begin(write=False) as txn:
            return txn.get(key)

    def num_keys(self) -> int:
        return self.env.stat()["entries"]

    def clear(self) -> bool:
        db = self.env.open_db()

        with self.env.begin(write=True) as txn:
            txn.drop(db)

        return self.num_keys()

    def mset(self, mapping: Dict[str, str]):
        kvpairs = list(zip(mapping.keys(), mapping.values()))

        with self.env.begin(write=True) as txn:
            cursor = txn.cursor()
            consumed_cnt, added_cnt = cursor.putmulti(kvpairs)

        if consumed_cnt != added_cnt:
            raise RuntimeError(
                "LmdbStore mset failed with: {}, failed to set {} items".format(
                    mapping, consumed_cnt - added_cnt
                )
            )

    def mget(self, keys: List[str]) -> List[Optional[str]]:

        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            kvpairs = cursor.getmulti(keys)

        mapping = {k: v for k, v in kvpairs}
        return list(map(lambda k: mapping.get(k, None), keys))

    def status(self) -> bool:
        return self.env.stat()

    def shutdown(self):
        self.env.close()
