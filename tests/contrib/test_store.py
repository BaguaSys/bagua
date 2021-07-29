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
import numpy as np
import pickle

logging.basicConfig(level=logging.DEBUG)


class TestLmdbStore(unittest.TestCase):
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
        store = LmdbStore(path=".lmdb", capacity_per_node=10000000, overwrite=True)
        self.check(store)


class TestRedisStore(unittest.TestCase):
    def check(self, store):
        self.generated_data = [np.random.rand(10) for _ in range(5)]
        store.set("1", pickle.dumps(self.generated_data[1]))

        store.mset(
            {
                "2": pickle.dumps(self.generated_data[2]),
                "3": pickle.dumps(self.generated_data[3]),
            }
        )
        ret = store.mget(["2", "4"])
        self.assertTrue((pickle.loads(ret[0]) == self.generated_data[2]).all())
        self.assertEqual(ret[1], None)

        r1 = store.get("1")
        r2 = store.get("4")
        self.assertTrue((pickle.loads(r1) == self.generated_data[1]).all())
        self.assertEqual(r2, None)

        cnt = store.num_keys()
        self.assertEqual(cnt, 3)

        store.clear()
        self.assertEqual(store.num_keys(), 0)

        self.assertTrue(store.status())

    def test_redis_store(self):
        store = RedisStore(hosts=None, cluster_mode=False, capacity_per_node=10000000)
        self.check(store)

        # try to shut down resources
        store.shutdown()

    def test_redis_cluster_store(self):
        return
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

        store = RedisStore(hosts=hosts, cluster_mode=True, capacity_per_node=10000000)
        self.check(store)

        self.assertTrue(store.status())

        # Now shut down servers safely
        for port in ports:
            client = redis.Redis(port=port)
            client.shutdown(nosave=True)
            client.close()


if __name__ == "__main__":
    unittest.main()
