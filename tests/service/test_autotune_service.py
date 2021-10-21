import unittest
import logging
import multiprocessing
import socket
import time
import torch.distributed as dist
from flask import Flask
from typing import List
from bagua.bagua_define import TensorDeclaration, BaguaCoreTelemetrySpan
from bagua.service import AutotuneService, AutotuneClient
from bagua.bagua_define import BaguaHyperparameter, get_tensor_declaration_bytes
from tests import skip_if_cuda_available


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
        score += -abs(
            sum([get_tensor_declaration_bytes(td) for td in bucket]) - 20 * 1024 ** 2
        )

    if not is_hierarchical_reduce:
        score += abs(score) * 0.1

    return score


class MockBaguaProcess:
    def __init__(
        self,
        rank: int,
        service_addr: str,
        service_port: int,
        model_name: str,
        tensor_list: List[TensorDeclaration],
        spans: List[BaguaCoreTelemetrySpan] = [],
    ) -> None:
        self.rank = rank
        self.model_name = model_name
        self.tensor_list = tensor_list
        self.spans = spans
        self.client = AutotuneClient(service_addr, service_port)

    def run(self, world_size, pg_init_method: str = "tcp://localhost:29501"):
        dist.init_process_group(
            backend=dist.Backend.GLOO,
            rank=self.rank,
            world_size=world_size,
            init_method=pg_init_method,
        )

        rsp = self.client.register_tensors(self.model_name, self.tensor_list)
        assert rsp.status_code == 200, "register_tensors failed, rsp={}".format(rsp)
        hp = BaguaHyperparameter().update(rsp.json()["recommended_hyperparameters"])

        train_iter = 0
        while True:
            dist.barrier()

            score = metrics(hp.buckets, hp.is_hierarchical_reduce)
            rsp = self.client.report_metrics(
                self.model_name, self.rank, train_iter, hp.dict(), score
            )
            assert rsp.status_code == 200, "report_metrics failed, rsp={}".format(rsp)
            rsp = self.client.report_tensor_execution_order(self.spans)
            assert (
                rsp.status_code == 200
            ), "report_tensor_execution_order failed, rsp={}".format(rsp)
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

            time.sleep(0.1)
            train_iter += 1

        return hp


class TestAutotuneService(unittest.TestCase):
    @skip_if_cuda_available()
    def test_autotune_service(self):
        service_addr = "127.0.0.1"
        service_port = pick_n_free_ports(1)[0]
        nprocs = 2

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
        log.setLevel(logging.INFO)

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
            "basic": (
                [
                    TensorDeclaration(
                        {
                            "name": "basic.A",
                            "num_elements": 1 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "basic.B",
                            "num_elements": 2 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "basic.C",
                            "num_elements": 3 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "basic.D",
                            "num_elements": 4 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "basic.E",
                            "num_elements": 5 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                ],
                [],
            ),
            "Mixed_precision_test": (
                [
                    TensorDeclaration(
                        {
                            "name": "Mixed_precision_test.A",
                            "num_elements": 1 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "Mixed_precision_test.B",
                            "num_elements": 3 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "Mixed_precision_test.C",
                            "num_elements": 5 * 1024 ** 2,
                            "dtype": "f16",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "Mixed_precision_test.D",
                            "num_elements": 7 * 1024 ** 2,
                            "dtype": "f16",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "Mixed_precision_test.E",
                            "num_elements": 11 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                ],
                [],
            ),
            "out_of_order_tensor": (
                [
                    TensorDeclaration(
                        {
                            "name": "out_of_order_tensor.A",
                            "num_elements": 1 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "out_of_order_tensor.B",
                            "num_elements": 2 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "out_of_order_tensor.C",
                            "num_elements": 3 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "out_of_order_tensor.D",
                            "num_elements": 4 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                    TensorDeclaration(
                        {
                            "name": "out_of_order_tensor.E",
                            "num_elements": 5 * 1024 ** 2,
                            "dtype": "f32",
                        }
                    ),
                ],
                [
                    {
                        "trace_id": 0,
                        "action": "tensor_ready",
                        "tensor_name": "out_of_order_tensor.D",
                        "start_time": 0,
                        "end_time": 1,
                    },
                    {
                        "trace_id": 1,
                        "action": "tensor_ready",
                        "tensor_name": "out_of_order_tensor.E",
                        "start_time": 1,
                        "end_time": 2,
                    },
                    {
                        "trace_id": 2,
                        "action": "tensor_ready",
                        "tensor_name": "out_of_order_tensor.A",
                        "start_time": 2,
                        "end_time": 3,
                    },
                    {
                        "trace_id": 3,
                        "action": "tensor_ready",
                        "tensor_name": "out_of_order_tensor.B",
                        "start_time": 3,
                        "end_time": 14,
                    },
                    {
                        "trace_id": 4,
                        "action": "tensor_ready",
                        "tensor_name": "out_of_order_tensor.C",
                        "start_time": 4,
                        "end_time": 5,
                    },
                ],
            ),
        }

        mock_objs = []
        pool = multiprocessing.Pool(nprocs * len(model_dict))
        results = dict([(key, []) for key in model_dict.keys()])
        for (model_name, (tensor_list, spans)) in model_dict.items():
            pg_init_method = "file:///tmp/.bagua.unittest.autotune.{}".format(model_name)
            for i in range(nprocs):
                mock = MockBaguaProcess(
                    i, service_addr, service_port, model_name,
                    tensor_list, spans
                )
                mock_objs.append(mock)
                ret = pool.apply_async(mock.run, (nprocs, pg_init_method, ))
                results[model_name].append(ret)

        pool.close()
        pool.join()

        for ret in results["basic"]:
            hp = ret.get()
            buckets = [[td["name"] for td in bucket] for bucket in hp.buckets]
            self.assertEqual(
                buckets,
                [["basic.A", "basic.B", "basic.C"], ["basic.D"], ["basic.E"]],
                "hp={}".format(hp.dict()),
            )
        for ret in results["Mixed_precision_test"]:
            hp = ret.get()
            buckets = [[td["name"] for td in bucket] for bucket in hp.buckets]
            self.assertEqual(
                buckets,
                [
                    ["Mixed_precision_test.C", "Mixed_precision_test.D"],
                    ["Mixed_precision_test.A", "Mixed_precision_test.B"],
                    ["Mixed_precision_test.E"],
                ],
                "hp={}".format(hp.dict()),
            )
        for ret in results["out_of_order_tensor"]:
            hp = ret.get()
            buckets = [[td["name"] for td in bucket] for bucket in hp.buckets]
            self.assertEqual(
                buckets,
                [
                    ["out_of_order_tensor.D"],
                    ["out_of_order_tensor.E"],
                    [
                        "out_of_order_tensor.A",
                        "out_of_order_tensor.B",
                        "out_of_order_tensor.C",
                    ],
                ],
                "hp={}".format(hp.dict()),
            )

        server.terminate()
        server.join()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARN,
        format="%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s",
    )
    unittest.main()
