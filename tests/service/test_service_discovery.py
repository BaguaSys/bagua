import unittest
from bagua.service import pick_n_free_ports, generate_and_broadcast_server_addr


class TestServiceDiscovery(unittest.TestCase):
    def test_generate_and_broadcast_server_addr(self):
        master_addr = "127.0.0.1"
        master_port = pick_n_free_ports(1)[0]

        import multiprocessing

        n_agent = 3
        result = list(range(n_agent))

        def get_addr(i):
            result[i] = generate_and_broadcast_server_addr(
                master_addr, master_port, n_agent, i
            )

        jobs = []
        for i in range(n_agent):
            p = multiprocessing.Process(target=get_addr, args=(i,))
            p.start()
            jobs.append(p)

        for p in jobs:
            p.join()

        for x in result:
            self.assertEqual(x, result[0])


if __name__ == "__main__":
    unittest.main()
