from typing import List, Dict, Optional
from collections import defaultdict


__all__ = ["Store", "ClusterStore"]


class Store:
    """
    Base class for all store implementations. A store keeps a mapping from keys to values.
    key-value pairs are manually added to store using `set()` or `mset()` and can be retrieved by
    `get()` or `mget()`.
    """

    def set(self, key: str, value: str):
        pass

    def get(self, key: str) -> Optional[str]:
        pass

    def num_keys(self) -> int:
        pass

    def clear(self):
        pass

    def mset(self, mapping: Dict[str, str]):
        pass

    def mget(self, keys: List[str]) -> List[Optional[str]]:
        pass

    def status(self) -> bool:
        pass

    def shutdown(self):
        pass


class ClusterStore(Store):
    """
    An implementation for a store cluster.

    This class implements client side sharding. It uses CRC-16 algorithm to compute the shard key by default, and can
    accept customized hashing algorithms by passing `hash_fn` on initialization.

    key-value pairs are manually added to the cluster using `set()` or `mset()` and can be retrieved by
    `get()` or `mget()`.

    """

    def __init__(self, stores: List[Store], hash_fn=None):

        self.stores = stores
        self.num_stores = len(stores)

        if hash_fn is None:
            from .hash_func import crc16

            hash_fn = crc16
        self.hash_fn = hash_fn

    def _hash_key(self, key) -> int:
        hash_code = self.hash_fn(key)
        return hash_code % self.num_stores

    def route(self, key) -> Store:
        return (
            self.stores[self._hash_key(key)] if self.num_stores > 1 else self.stores[0]
        )

    def set(self, key: str, value: str):
        if self.num_stores == 1:
            return self.stores[0].set(key, value)

        self.route(key).set(key, value)

    def get(self, key: str) -> Optional[str]:
        if self.num_stores == 1:
            return self.stores[0].get(key)

        return self.route(key).get(key)

    def num_keys(self) -> int:
        return sum([store.num_keys() for store in self.stores])

    def clear(self):
        for store in self.stores:
            store.clear()

    def mset(self, mapping: Dict[str, str]):
        if self.num_stores == 1:
            return self.stores[0].mset(mapping)

        route_table = {}
        for k, v in mapping.items():
            sid = self._hash_key(k)
            m = route_table.get(sid, defaultdict(dict))
            m[k] = v
            route_table[sid] = m

        for sid, m in route_table.items():
            self.stores[sid].mset(m)

    def mget(self, keys: List[str]) -> List[Optional[str]]:
        if self.num_stores == 1:
            return self.stores[0].mget(keys)

        route_table = {}
        for k in keys:
            sid = self._hash_key(k)
            ll = route_table.get(sid, [])
            ll.append(k)
            route_table[sid] = ll

        result_map = {}
        for sid, ll in route_table.items():
            ret = self.stores[sid].mget(ll)
            m = {k: v for k, v in zip(ll, ret)}
            result_map = {**result_map, **m}

        return list(map(lambda x: result_map.get(x, None), keys))

    def status(self) -> bool:
        return all([store.status() for store in self.stores])

    def shutdown(self):
        for store in self.stores:
            store.shutdown()
