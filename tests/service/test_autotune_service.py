import unittest
import logging
import multiprocessing
from flask import Flask
from bagua.bagua_define import TensorDeclaration
from bagua.service import AutotuneService, AutotuneClient
from bagua.bagua_define import BaguaHyperparameter, get_tensor_declaration_bytes
import socket
import numpy as np


def pick_n_free_ports(n: int):
    socks = []
    for i in range(n):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        socks.append(sock)

    n_free_ports = [sock.getsockname()[1] for sock in socks]
    for sock in socks:
        sock.close()

    return n_free_ports


def metrics(buckets, is_hierarchical_reduce):
    # score = sum(-abs(bucket_sum_size - 20MB))
    # convex function with peak at bucket_size=20MB
    score = 0.0
    for bucket in buckets:
        score += -abs(sum([get_tensor_declaration_bytes(td)
                      for td in bucket]) - 20 * 1024 ** 2)

    if not is_hierarchical_reduce:
        score += abs(score) * 0.1

    return score


class MockBaguaProcess:
    def __init__(
        self,
        rank,
        service_addr,
        service_port,
        model_name,
        tensor_list,
    ) -> None:
        self.rank = rank
        self.model_name = model_name
        self.tensor_list = tensor_list
        self.client = AutotuneClient(service_addr, service_port)

    def run(self, total_iters=5000):
        rsp = self.client.register_tensors(self.model_name, self.tensor_list)
        assert rsp.status_code == 200, "register_tensors failed, rsp={}".format(
            rsp)
        hp = BaguaHyperparameter().update(
            rsp.json()["recommended_hyperparameters"])

        for train_iter in range(total_iters):
            score = metrics(hp.buckets, hp.is_hierarchical_reduce)
            rsp = self.client.report_metrics(
                self.model_name, self.rank, train_iter, hp.dict(), score
            )
            assert rsp.status_code == 200, "report_metrics failed, rsp={}".format(
                rsp)
            rsp = self.client.ask_hyperparameters(
                self.model_name, self.rank, train_iter
            )
            assert rsp.status_code == 200, "ask_hyperparameters failed, rsp={}".format(
                rsp
            )
            rsp = rsp.json()
            hp.update(rsp["recommended_hyperparameters"])

            if rsp["is_autotune_completed"]:
                logging.info("train_iter={}".format(train_iter))
                break

        return hp


class TestAutotuneService(unittest.TestCase):
    def test_autotune_service(self):
        service_addr = "127.0.0.1"
        service_port = pick_n_free_ports(1)[0]
        nprocs = 2
        np.random.seed(123)

        autotune_service = AutotuneService(
            nprocs,
            autotune_level=1,
            sampling_confidence_time_s=0.1,
            warmup_time_s=1.0,
            is_output_autotune_log=True,
        )
        app = Flask(__name__)
        app = autotune_service.setup_app(app)
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        server = multiprocessing.Process(
            target=app.run,
            kwargs={
                "host": "0.0.0.0",
                "port": service_port,
                "debug": False,
            },
        )
        server.daemon = True
        server.start()

        model_dict = {
            "m1": [
                TensorDeclaration(
                    {
                        "name": "A",
                        "num_elements": 1 * 1024 ** 2,
                        "dtype": "f32",
                    }
                ),
                TensorDeclaration(
                    {
                        "name": "B",
                        "num_elements": 2 * 1024 ** 2,
                        "dtype": "f32",
                    }
                ),
                TensorDeclaration(
                    {
                        "name": "C",
                        "num_elements": 3 * 1024 ** 2,
                        "dtype": "f32",
                    }
                ),
                TensorDeclaration(
                    {
                        "name": "D",
                        "num_elements": 4 * 1024 ** 2,
                        "dtype": "f32",
                    }
                ),
                TensorDeclaration(
                    {
                        "name": "E",
                        "num_elements": 5 * 1024 ** 2,
                        "dtype": "f32",
                    }
                ),
            ],
            "m2": [
                TensorDeclaration(
                    {
                        "name": "A",
                        "num_elements": 1 * 1024 ** 2,
                        "dtype": "f32",
                    }
                ),
                TensorDeclaration(
                    {
                        "name": "B",
                        "num_elements": 3 * 1024 ** 2,
                        "dtype": "f32",
                    }
                ),
                TensorDeclaration(
                    {
                        "name": "C",
                        "num_elements": 5 * 1024 ** 2,
                        "dtype": "f16",
                    }
                ),
                TensorDeclaration(
                    {
                        "name": "D",
                        "num_elements": 7 * 1024 ** 2,
                        "dtype": "f16",
                    }
                ),
                TensorDeclaration(
                    {
                        "name": "E",
                        "num_elements": 11 * 1024 ** 2,
                        "dtype": "f32",
                    }
                ),
            ],
        }

        mock_objs = []
        with multiprocessing.pool.ThreadPool(nprocs * len(model_dict)) as pool:
            results = dict([(key, []) for key in model_dict.keys()])
            for i in range(nprocs):
                for (model_name, tensor_list) in model_dict.items():
                    mock = MockBaguaProcess(
                        i, service_addr, service_port, model_name, tensor_list
                    )
                    mock_objs.append(mock)
                    ret = pool.apply_async(mock.run)
                    results[model_name].append(ret)
            for ret in results["m1"]:
                hp = ret.get()
                buckets = [[
                    td["name"] for td in bucket] for bucket in hp.buckets]
                self.assertEqual(
                    buckets,
                    [['A', 'B', 'C'], ['D'], ['E']],
                    "hp={}".format(hp.dict())
                )
            for ret in results["m2"]:
                hp = ret.get()
                buckets = [[
                    td["name"] for td in bucket] for bucket in hp.buckets]
                self.assertEqual(
                    buckets,
                    [['C', 'D'], ['A', 'B'], ['E']],
                    "hp={}".format(hp.dict())
                )

        server.terminate()
        server.join()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARN,
        format="%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s",
    )
    unittest.main()
