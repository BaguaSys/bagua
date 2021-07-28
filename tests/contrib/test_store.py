import unittest
from bagua.torch_api.contrib.utils.redis_store import (
    RedisStore,
    start_redis_server_cli,
    create_redis_cluster_cli,
    find_free_port,
)
from bagua.torch_api.contrib.utils.lmdb_store import LmdbStore
import redis
import multiprocessing as mp
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

        cnt = store.num_keys()
        self.assertEqual(cnt, 5)

        store.clear()
        self.assertEqual(store.num_keys(), 0)

        self.assertTrue(store.status())

        # shut down resources at the end
        store.shutdown()

    def test_lmdb_store(self):
        store = LmdbStore(name=".test.lmdb", overwrite=True)
        self.check(store)

    def test_redis_store(self):
        store = RedisStore(bootstrap=True)
        self.check(store)


class TestClusterStore(unittest.TestCase):
    def check(self, store):
        store.set("a", 1)

        store.mset({"b": 2, "c": 3})
        ret = store.mget(["b", "d"])
        self.assertEqual(ret[0], str(2))
        self.assertEqual(ret[1], None)

        r1 = store.get("a")
        r2 = store.get("d")
        self.assertEqual(r1, str(1))
        self.assertEqual(r2, None)

        cnt = store.num_keys()
        self.assertEqual(cnt, 3)

        store.clear()
        self.assertEqual(store.num_keys(), 0)

        self.assertTrue(store.status())

        # try to shut down resources
        store.shutdown()

        self.assertTrue(store.status())

    def test_redis_cluster_store(self):
        n = 3
        hosts = []
        ports = []
        processes = []
        for i in range(n):
            port = find_free_port()
            p = mp.Process(
                target=start_redis_server_cli,
                args=(port, True, 10000000, f"--cluster-config-file nodes{port}.conf"),
            )
            p.start()

            ports.append(port)
            processes.append(p)
            hosts.append({"host": "127.0.0.1", "port": port})

        for p in processes:
            p.join()

        create_redis_cluster_cli(hosts=hosts)
        store = RedisStore(hosts=hosts, bootstrap=False, overwrite=True)
        self.check(store)

        # Now shut down servers safely
        for port in ports:
            client = redis.Redis(port=port)
            client.shutdown(nosave=True)
            client.close()


if __name__ == "__main__":
    unittest.main()
