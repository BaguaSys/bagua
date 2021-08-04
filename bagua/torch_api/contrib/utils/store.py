from typing import List, Dict, Optional, Any
from .hash_func import crc16
from collections import defaultdict


class Store:
    def set(self, key: str, value: str):
        pass

    def get(self, key: str) -> Optional[str]:
        pass

    def num_keys(self) -> int:
        pass

    def clear(self) -> bool:
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
    def __init__(self, stores: List[Store]):
        self.stores = stores
        self.num_stores = len(stores)

    def _hash_key(self, key):
        hash_code = crc16(key)
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

    def clear(self) -> bool:
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
            l = route_table.get(sid, [])
            l.append(k)
            route_table[sid] = l

        result_map = {}
        for sid, l in route_table.items():
            ret = self.stores[sid].mget(l)
            m = {k: v for k, v in zip(l, ret)}
            result_map = {**result_map, **m}

        return list(map(lambda x: result_map.get(x, None), keys))

    def status(self) -> bool:
        return all([store.status() for store in self.stores])

    def shutdown(self):
        for store in self.stores:
            store.shutdown()
