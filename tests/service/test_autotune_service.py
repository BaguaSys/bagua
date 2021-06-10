import unittest
import logging
import requests
import time
import multiprocessing
import numpy as np
from flask import Flask
from bagua.autotune import AutoTuneHyperParams
from bagua.service import AutotuneService, BaguaHyperparameter, pick_n_free_ports


def metrics(buckets, is_hierarchical_reduce):
    # score = sum(-abs(bucket_sum_size - 5M))
    # 峰在 bucket_size=5M 的凸函数
    score = 0.0
    for bucket in buckets:
        score += -abs(sum([td["num_elements"] for td in bucket]) - 5 * 1024 ** 2)

    if not is_hierarchical_reduce:
        score += abs(score) * 0.1

    return score


class AutotuneClient:
    def __init__(self, autotune_service_addr):
        self.autotune_service_addr = autotune_service_addr

    def report_metrics(
        self,
        rank,
        unix_timestamp,
        train_iter,
        iter_per_seconds,
        denoised_iter_per_seconds,
        hyperparameters,
    ):
        try:
            rsp = requests.post(
                "http://{}/api/v1/report_metrics".format(self.autotune_service_addr),
                json={
                    "rank": rank,
                    "unix_timestamp": unix_timestamp,
                    "train_iter": train_iter,
                    "iter_per_seconds": iter_per_seconds,
                    "denoised_iter_per_seconds": denoised_iter_per_seconds,
                    "hyperparameters": hyperparameters,
                },
            )
            return rsp.status_code
        except Exception as ex:
            logging.warning(
                "rank={}, train_iter={}, ex={}".format(rank, train_iter, ex)
            )
            return -1

    def request_checkboard(self):
        while True:
            try:
                rsp = requests.get(
                    "http://{}/api/v1/checkboard".format(self.autotune_service_addr)
                )
                return rsp
            except Exception as ex:
                logging.warning("ex={}".format(ex))
                continue

    def wait_for_all_process_parameters_updated(self, my_train_iter):
        while True:
            rsp = self.request_checkboard()
            assert rsp.status_code == 200, "rsp={}".format(rsp)
            check_board = rsp.json()["check_board"]
            if all([train_iter >= my_train_iter for train_iter in check_board]):
                break
            time.sleep(0)

    def ask_hyperparameters(self, rank, train_iter):
        while True:
            try:
                rsp = requests.post(
                    "http://{}/api/v1/ask_hyperparameters".format(
                        self.autotune_service_addr
                    ),
                    json={
                        "rank": rank,
                        "train_iter": train_iter,
                    },
                )
                return rsp
            except Exception as ex:
                logging.warning("rank={}, exception={}".format(rank, ex))
                time.sleep(1)
                continue


class MockBaguaProcess:
    def __init__(self, rank, autotune_service_addr):
        self.rank = rank
        self.iter_per_seconds = 0.0
        self.denoised_iter_per_seconds = 0.0
        self.hyperparameters = BaguaHyperparameter(
            buckets=[
                [
                    {
                        "name": "A",
                        "num_elements": 1 * 1024 ** 2,
                        "dtype": "F32",
                    }
                ],
                [
                    {
                        "name": "B",
                        "num_elements": 3 * 1024 ** 2,
                        "dtype": "F32",
                    }
                ],
                [
                    {
                        "name": "C",
                        "num_elements": 5 * 1024 ** 2,
                        "dtype": "F32",
                    }
                ],
                [
                    {
                        "name": "D",
                        "num_elements": 7 * 1024 ** 2,
                        "dtype": "F32",
                    }
                ],
                [
                    {
                        "name": "E",
                        "num_elements": 11 * 1024 ** 2,
                        "dtype": "F32",
                    }
                ],
            ],
            is_hierarchical_reduce=False,
        )
        self.client = AutotuneClient(autotune_service_addr)

    def run(self, total_iters=1200):
        for i in range(total_iters):
            if i != 0 and i % 10 == 0:
                rsp = self.client.ask_hyperparameters(self.rank, i)
                assert rsp.status_code == 200, "rsp={}".format(rsp)
                self.hyperparameters.update(rsp.json()["recommended_hyperparameters"])
                self.client.wait_for_all_process_parameters_updated(i)

            score = metrics(
                self.hyperparameters.buckets,
                self.hyperparameters.is_hierarchical_reduce,
            )
            logging.info(
                "train_iter={}, score={}, hyperparameters={}".format(
                    i, score, self.hyperparameters.dict()
                )
            )
            ret = self.client.report_metrics(
                self.rank, time.time(), i, score, score, self.hyperparameters.dict()
            )
            assert ret == 200, "ret={}".format(ret)

        score = metrics(
            self.hyperparameters.buckets,
            self.hyperparameters.is_hierarchical_reduce,
        )
        logging.info(
            "best score={}, best hyperparameters={}".format(
                score, self.hyperparameters.dict()
            )
        )
        assert score == -8493465.6, "score={}".format(score)

        return score


class TestAutotuneService(unittest.TestCase):
    def test_autotune_service(self):
        master_addr = "127.0.0.1"
        master_port = pick_n_free_ports(1)[0]
        autotune_service_addr = "{}:{}".format(master_addr, master_port)
        nprocs = 2

        autotune_service = AutotuneService(
            nprocs, max_samples=100, sampling_confidence_time_s=0
        )
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

        mock_objs = []
        with multiprocessing.pool.ThreadPool(nprocs) as pool:
            results = []
            for i in range(nprocs):
                mock = MockBaguaProcess(i, autotune_service_addr)
                mock_objs.append(mock)
                ret = pool.apply_async(mock.run)
                results.append(ret)
            for ret in results:
                score = ret.get()
                self.assertEqual(score, -8493465.6)

        server.terminate()
        server.join()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s",
    )
    unittest.main()
