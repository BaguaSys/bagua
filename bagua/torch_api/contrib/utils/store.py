from typing import List, Dict, Optional, Union
from collections import defaultdict


__all__ = ["Store", "ClusterStore"]


class Store:
    """
    Base class for key-value store implementations. Entries are added to store with :meth:`set` or :meth:`mset`, and retrieved
    with :meth:`get` or :meth:`mget`.
    """

    def set(self, key: str, value: Union[str, bytes]):
        """Set a key-value pair."""
        pass

    def get(self, key: str) -> Optional[Union[str, bytes]]:
        """Returns the value associated with :attr:`key`, or ``None`` if the key doesn't exist."""
        pass  # type: ignore

    def num_keys(self) -> int:
        """Returns the number of keys in the current store."""
        pass  # type: ignore

    def clear(self):
        """Delete all keys in the current store."""
        pass

    def mset(self, dictionary: Dict[str, Union[str, bytes]]):
        """
        Set multiple entries at once with a dictionary. Each key-value pair in the :attr:`dictionary` will be set.
        """
        pass

    def mget(self, keys: List[str]) -> List[Optional[Union[str, bytes]]]:
        """
        Retrieve each key's corresponding value and return them in a list with the same order as :attr:`keys`.
        """

        pass  # type: ignore

    def status(self) -> bool:
        """
        Returns ``True`` if the current store is alive.
        """
        pass  # type: ignore

    def shutdown(self):
        """
        Shutdown the managed store instances. Unmanaged instances will not be killed.
        """
        pass


class ClusterStore(Store):
    """
    Base class for distributed key-value stores.

    In cluster store, entries will be sharded equally among multiple store instances based on their keys.

    Args:
        stores(List[Store]): A list of stores to shard entries on.

    """

    def __init__(self, stores: List[Store]):

        self.stores = stores
        self.num_stores = len(stores)

        import xxhash

        def xxh64(x):
            return xxhash.xxh64(x).intdigest()

        self.hash_fn = xxh64

    def _hash_key(self, key: str) -> int:
        hash_code = self.hash_fn(key.encode())
        return hash_code % self.num_stores

    def route(self, key: str) -> Store:
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

    def mset(self, dictionary: Dict[str, Union[str, bytes]]):
        if self.num_stores == 1:
            return self.stores[0].mset(dictionary)

        route_table = {}
        for k, v in dictionary.items():
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
