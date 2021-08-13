from typing import List, Dict, Optional, Union
from collections import defaultdict


__all__ = ["Store", "ClusterStore"]


class Store:
    """
    Base class for all Key-Value store implementations. A store keeps a mapping from keys to values.
    key-value pairs are manually added to store using `set()` or `mset()` and can be retrieved by
    `get()` or `mget()`.
    """

    def set(self, key, value):
        """Set a key-value pair."""
        pass

    def get(self, key) -> Optional[Union[str, bytes]]:
        """Returns the value associated with key `key`, or None if the key doesn't exist."""
        pass  # type: ignore

    def num_keys(self) -> int:
        """Returns the number of keys in the current store."""
        pass  # type: ignore

    def clear(self):
        """Delete all keys in the current store."""
        pass

    def mset(self, mapping):
        """
        Set key/values based on a mapping. Mapping is a dictionary of key/value pairs.
        """
        pass

    def mget(self, keys) -> List[Optional[Union[str, bytes]]]:
        """
        Returns a list of values ordered identically to `keys`.
        """

        pass  # type: ignore

    def status(self) -> bool:
        """
        Returns the status of the current store.
        """
        pass  # type: ignore

    def shutdown(self):
        """
        Shutdown the current store. External store resources, for example, initialized redis servers,
        will not be shut down by this method.
        """
        pass


class ClusterStore(Store):
    """
    Base class for distributed Key-Value stores.

    This class implements client side sharding. It uses **xxHash** algorithm to compute the shard key by default, and can
    accept customized hashing algorithms by passing `hash_fn` on initialization.

    key-value pairs are manually added to the cluster using `set()` or `mset()` and can be retrieved by
    `get()` or `mget()`.

    Args:
        stores(List[Store]): A list of stores in the cluster.
        hash_fn: Hash function to compute the shard key. Default is `xxh64`. A `hash_fn` accepts a `str` as
            input, and returns an `int` as output.

    """

    def __init__(self, stores: List[Store], hash_fn=None):

        self.stores = stores
        self.num_stores = len(stores)

        if hash_fn is None:
            import xxhash

            def xxh64(x):
                return xxhash.xxh64(x).intdigest()

            hash_fn = xxh64

        self.hash_fn = hash_fn

    def _hash_key(self, key) -> int:
        hash_code = self.hash_fn(key)
        return hash_code % self.num_stores

    def route(self, key) -> Store:
        return (
            self.stores[self._hash_key(key)] if self.num_stores > 1 else self.stores[0]
        )

    def set(self, key: str, value: Union[str, bytes]):
        if self.num_stores == 1:
            return self.stores[0].set(key, value)

        self.route(key).set(key, value)

    def get(self, key: str) -> Optional[Union[str, bytes]]:
        if self.num_stores == 1:
            return self.stores[0].get(key)

        return self.route(key).get(key)

    def num_keys(self) -> int:
        return sum([store.num_keys() for store in self.stores])

    def clear(self):
        for store in self.stores:
            store.clear()

    def mset(self, mapping: Dict[str, Union[str, bytes]]):
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

    def mget(self, keys: List[str]) -> List[Optional[Union[str, bytes]]]:
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
