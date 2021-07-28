import lmdb
from .store import Store
from typing import List, Dict, Optional


class LmdbStore(Store):
    def __init__(self, name, map_size: int = 1_000_000_000):
        self.map_size = map_size
        self.name = name
        self.db = lmdb.open(self.name, map_size=self.map_size)

    def set(self, key: str, value: str):
        with self.db.begin(write=True) as txn:
            txn.put(key, value)

    def get(self, key: str) -> Optional[str]:
        with self.db.begin(write=False) as txn:
            return txn.get(key)

    def num_keys(self) -> int:
        return self.db.stat()["entries"]

    def clear(self):
        # TODO
        raise NotImplementedError("not implemented in `LmdbStore`")

    def mset(self, mapping: Dict[str, str]):
        kvpairs = list(zip(mapping.keys(), mapping.values()))

        with self.db.begin(write=True) as txn:
            cursor = txn.cursor()
            consumed_cnt, added_cnt = cursor.putmulti(kvpairs)

        if consumed_cnt != added_cnt:
            raise RuntimeError(
                "LmdbStore mset failed with: {}, failed to set {} items".format(
                    mapping, consumed_cnt - added_cnt
                )
            )

    def mget(self, keys: List[str]) -> List[Optional[str]]:

        with self.db.begin(write=False) as txn:
            cursor = txn.cursor()
            kvpairs = cursor.getmulti(keys)

        mapping = {k: v for k, v in kvpairs}
        return list(map(lambda k: mapping.get(k, None), keys))

    def status(self) -> bool:
        # TODO
        raise NotImplementedError("not implemented in `LmdbStore`")
