import unittest
from bagua.torch_api.contrib.utils.redis_store import (
    RedisStore,
    start_redis_server_cli,
    find_free_port,
)
import redis
import multiprocessing as mp
import logging
import numpy as np
import pickle
from tests import skip_if_cuda_available


logging.basicConfig(level=logging.DEBUG)


class TestRedisStore(unittest.TestCase):
    def check(self, store):
        store.clear()
        self.assertEqual(store.num_keys(), 0)

        self.generated_data = [np.random.rand(10) for _ in range(5)]
        store.set("1", pickle.dumps(self.generated_data[1]))

        store.mset(
            {
                "2": pickle.dumps(self.generated_data[2]),
                "3": pickle.dumps(self.generated_data[3]),
                "4": pickle.dumps(self.generated_data[4]),
            }
        )
        ret = store.mget(["1", "2", "5"])
        self.assertTrue((pickle.loads(ret[0]) == self.generated_data[1]).all())
        self.assertTrue((pickle.loads(ret[1]) == self.generated_data[2]).all())
        self.assertEqual(ret[2], None)

        r1 = store.get("4")
        r2 = store.get("6")
        self.assertTrue((pickle.loads(r1) == self.generated_data[4]).all())
        self.assertEqual(r2, None)

        cnt = store.num_keys()
        self.assertEqual(cnt, 4)

        self.assertTrue(store.status())

    @skip_if_cuda_available()
    def test_redis_store(self):
        store = RedisStore(hosts=None, cluster_mode=False, capacity_per_node=10000000)
        self.check(store)

    @skip_if_cuda_available()
    def test_redis_cluster_store(self):
        n = 3
        hosts = []
        ports = []
        processes = []
        for i in range(n):
            port = find_free_port()
            p = mp.Process(
                target=start_redis_server_cli,
                args=(port, 10000000, f"--cluster-config-file nodes{port}.conf"),
            )
            p.start()

            ports.append(port)
            processes.append(p)
            hosts.append({"host": "127.0.0.1", "port": port})

        for p in processes:
            p.join()
            self.assertTrue(p.exitcode == 0)

        store = RedisStore(hosts=hosts, cluster_mode=True)
        self.check(store)

        # Shut down servers manually
        for port in ports:
            client = redis.Redis(port=port)
            client.shutdown(nosave=True)
            client.close()


if __name__ == "__main__":
    unittest.main()
