import unittest
from bagua.torch_api.contrib.utils.redis_store import (
    RedisStore,
    start_redis_server_cli,
    create_redis_cluster_cli,
    find_free_port,
)
from bagua.torch_api.contrib.utils.lmdb_store import LmdbStore
import logging

logging.basicConfig(level=logging.DEBUG)


class TestStore(unittest.TestCase):
    def check(self, store):
        store.set(b"Beijing", b"China")
        store.set(b"Paris", b"France")

        store.mset({b"New Delhi": b"India", b"Tokyo": b"Japan", b"Madrid": b"Spain"})
        ret = store.mget([b"Beijing", b"London", b"Tokyo"])
        self.assertEqual(ret[0], b"China")
        self.assertEqual(ret[1], None)
        self.assertEqual(ret[2], b"Japan")

        r1 = store.get(b"Madrid")
        r2 = store.get(b"Shanghai")
        self.assertEqual(r1, b"Spain")
        self.assertEqual(r2, None)

    def test_redis_store(self):
        store = RedisStore(bootstrap=True)
        self.check(store)

    def test_redis_cluster_store(self):
        n = 3
        hosts = []
        for i in range(n):
            port = find_free_port()
            start_redis_server_cli(
                port, True, f"--cluster-config-file nodes{port}.conf"
            )
            hosts.append({"host": "127.0.0.1", "port": port})

        create_redis_cluster_cli(hosts=hosts)
        store = RedisStore(hosts=hosts, bootstrap=False)
        self.check(store)

    def test_lmdb_store(self):
        store = LmdbStore(name=".test.lmdb")
        self.check(store)


if __name__ == "__main__":
    unittest.main()
