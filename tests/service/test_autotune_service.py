import unittest
import logging
import requests
import time
import copy
import multiprocessing
import numpy as np
from flask import Flask
from bagua.bagua_define import TensorDeclaration
from bagua.service import (
    AutotuneService,
    AutotuneClient,
    BaguaHyperparameter,
)
from bagua.bagua_define import BaguaHyperparameter
import socket


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
    # score = sum(-abs(bucket_sum_size - 5M))
    # convex function with peak at bucket_size=5M
    score = 0.0
    for bucket in buckets:
        score += -abs(sum([td["num_elements"] for td in bucket]) - 5 * 1024 ** 2)

    if not is_hierarchical_reduce:
        score += abs(score) * 0.1

    return score


class MockBaguaProcess:
    def __init__(
        self, rank, autotune_service_addr,
        model_name, tensor_list,
    ) -> None:
        self.rank = rank
        self.model_name = model_name
        self.tensor_list = tensor_list
        self.client = AutotuneClient(autotune_service_addr)

    def run(self, total_iters=1200):
        rsp = self.client.register_tensors(
            self.model_name, self.tensor_list)
        assert rsp.status_code == 200, \
            "register_tensors failed, rsp={}".format(rsp)
        hp = BaguaHyperparameter().update(
            rsp.json()["recommended_hyperparameters"])

        for train_iter in range(total_iters):
            score = metrics(hp.buckets, hp.is_hierarchical_reduce)
            rsp = self.client.report_metrics(
                self.model_name, self.rank, train_iter, hp.dict(), score)
            assert rsp.status_code == 200, "report_metrics failed, rsp={}".format(rsp)
            rsp = self.client.ask_hyperparameters(
                self.model_name, self.rank, train_iter)
            assert rsp.status_code == 200, "ask_hyperparameters failed, rsp={}".format(rsp)
            hp.update(rsp["recommended_hyperparameters"])
            hp.update(rsp)

            if rsp["is_autotune_completed"]:
                logging.info("train_iter={}".format(train_iter))
                break

        return hp


class TestAutotuneService(unittest.TestCase):
    def test_autotune_service(self):
        master_addr = "127.0.0.1"
        master_port = pick_n_free_ports(1)[0]
        autotune_service_addr = "{}:{}".format(master_addr, master_port)
        nprocs = 2

        autotune_service = AutotuneService(nprocs, autotune_level=1)
        app = Flask(__name__)
        app = autotune_service.setup_app(app)

        server = multiprocessing.Process(
            target=app.run,
            kwargs={
                "host": "0.0.0.0",
                "port": master_port,
                "debug": False,
            },
        )
        server.daemon = True
        server.start()

        model_dict = {
            "m1": [
                TensorDeclaration({
                    "name": "A",
                    "num_elements": 1 * 1024 ** 2,
                    "dtype": "F32",
                }),
                TensorDeclaration({
                    "name": "B",
                    "num_elements": 2 * 1024 ** 2,
                    "dtype": "F32",
                }),
                TensorDeclaration({
                    "name": "C",
                    "num_elements": 3 * 1024 ** 2,
                    "dtype": "F32",
                }),
                TensorDeclaration({
                    "name": "D",
                    "num_elements": 4 * 1024 ** 2,
                    "dtype": "F32",
                }),
                TensorDeclaration({
                    "name": "E",
                    "num_elements": 5 * 1024 ** 2,
                    "dtype": "F32",
                }),
            ],
            "m2": [
                TensorDeclaration({
                    "name": "A",
                    "num_elements": 1 * 1024 ** 2,
                    "dtype": "F32",
                }),
                TensorDeclaration({
                    "name": "B",
                    "num_elements": 3 * 1024 ** 2,
                    "dtype": "F32",
                }),
                TensorDeclaration({
                    "name": "C",
                    "num_elements": 5 * 1024 ** 2,
                    "dtype": "F32",
                }),
                TensorDeclaration({
                    "name": "D",
                    "num_elements": 7 * 1024 ** 2,
                    "dtype": "F32",
                }),
                TensorDeclaration({
                    "name": "E",
                    "num_elements": 11 * 1024 ** 2,
                    "dtype": "F32",
                }),
            ],
        }

        mock_objs = []
        with multiprocessing.pool.ThreadPool(nprocs * len(model_dict)) as pool:
            results = []
            for i in range(nprocs):
                for (model_name, tensor_list) in model_dict.items():
                    mock = MockBaguaProcess(
                        i, autotune_service_addr, model_name, tensor_list)
                    mock_objs.append(mock)
                    ret = pool.apply_async(mock.run)
                    results.append(ret)
            for ret in results:
                hp = ret.get()
                print('hp={}'.format(hp))

        server.terminate()
        server.join()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s",
    )
    unittest.main()
