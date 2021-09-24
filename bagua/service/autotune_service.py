import copy
import requests
import os
import time
import threading
import json
import logging
import multiprocessing
from .autotune_task_manager import AutotuneTaskManager
from bagua.bagua_define import (
    TensorDtype,
    TensorDeclaration,
    BaguaCoreTelemetrySpan,
    BaguaHyperparameter,
)
from flask import request
import numpy as np
from typing import Dict, List


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, TensorDtype):
            return obj.value
        else:
            return super(NpEncoder, self).default(obj)


class AutotuneServiceTaskManager:
    def __init__(
        self, task_name: str, world_size: int, is_output_autotune_log: bool
    ) -> None:
        self.inner = AutotuneTaskManager(task_name, is_output_autotune_log)
        self.warmup_pass_count = 0
        self.sampling_count = 0
        self.lock = threading.Lock()
        self.check_board = [-1] * world_size
        self.time_hp_last_granted = time.time()
        self.hyperparameter = BaguaHyperparameter()


class AutotuneService:
    MAX_TRACE_INFO = 1000

    def __init__(
        self,
        world_size,
        autotune_level=0,
        max_samples=60,
        sampling_confidence_time_s=5,
        warmup_time_s=30,
        is_output_autotune_log=False,
        default_bucket_size=10 * 1024 ** 2,
    ):
        self.autotune_level = autotune_level
        self.world_size = world_size
        self.max_samples = max_samples
        self.sampling_confidence_time_s = sampling_confidence_time_s
        self.warmup_time_s = warmup_time_s
        self.is_initialized = False
        self.is_output_autotune_log = is_output_autotune_log
        self.default_bucket_size: int = default_bucket_size
        self.model_dict: Dict[str, AutotuneServiceTaskManager] = {}
        self.model_dict_mutex = threading.Lock()

        # bagua-core trace and obtain tensor calculation partial order
        self.trace_info_dict = {}
        self.tensor_partial_order = {}
        self.tensor_partial_order_fixed = False
        self.tensor_partial_order_lock = threading.Lock()

    def autotune(
        self,
        hp_manager: AutotuneServiceTaskManager,
        rank: int,
        train_iter: int,
        tensor_partial_order: Dict[str, int] = {},
    ):
        if hp_manager.sampling_count > self.max_samples:
            return

        (
            recommended_train_iter,
            hp,
            system_efficiency_score,
        ) = hp_manager.inner.tail_record()

        logging.info(
            "recommended_train_iter={}, hyperparameters={}, speed={}".format(
                recommended_train_iter,
                hp,
                system_efficiency_score,
            )
        )

        sampling_time = time.time() - hp_manager.time_hp_last_granted

        # Skip at least once during warmup
        if (
            sampling_time < self.warmup_time_s
            or hp_manager.warmup_pass_count == 0  # noqa: W503
        ):
            logging.info(
                "warmup pass, time.time={}, time_hp_last_granted={}, "
                "warmup_time_s={}".format(
                    time.time(),
                    hp_manager.time_hp_last_granted,
                    self.warmup_time_s,
                )
            )
            hp_manager.warmup_pass_count += 1
            return

        if hp_manager.sampling_count == 0:
            confidence_skip = (
                sampling_time < self.warmup_time_s + self.sampling_confidence_time_s
            )
        else:
            confidence_skip = sampling_time < self.sampling_confidence_time_s

        if confidence_skip:
            logging.debug(
                "The sampling time is not up, time={}, last={}, "
                "sampling_confidence_time_s={}".format(
                    time.time(),
                    hp_manager.time_hp_last_granted,
                    self.sampling_confidence_time_s,
                )
            )
            return

        logging.info(
            "rank={}, train_iter={}, sampling_count={}, "
            "max_samples={}".format(
                rank, train_iter, hp_manager.sampling_count, self.max_samples
            )
        )
        recommended_bagua_hp = hp_manager.inner.ask_hyperparmeter(
            train_iter, tensor_partial_order
        )
        if hp_manager.sampling_count < self.max_samples:
            hp_manager.hyperparameter = recommended_bagua_hp
        else:
            hp_manager.hyperparameter = hp_manager.inner.best_hyperparameter()

        hp_manager.sampling_count += 1

    def setup_app(self, app):
        @app.route("/api/v1/register_tensors", methods=["POST"])
        def register_tensors():
            req: dict = request.get_json(force=True)
            model_name: str = req["model_name"]
            tensor_list: List[TensorDeclaration] = req["tensor_list"]
            whether_to_bucket: bool = req["whether_to_bucket"]

            with self.model_dict_mutex:
                if model_name not in self.model_dict:
                    self.model_dict[model_name] = AutotuneServiceTaskManager(
                        task_name=model_name,
                        world_size=self.world_size,
                        is_output_autotune_log=self.is_output_autotune_log,
                    )

            hp_manager = self.model_dict[model_name]
            bucket_size = self.default_bucket_size
            if whether_to_bucket is False:
                bucket_size = 10 * 1024 ** 5

            with hp_manager.lock:
                hp = BaguaHyperparameter(
                    buckets=AutotuneTaskManager.split_bucket_by_bucket_size(
                        tensor_list,
                        bucket_size,
                    ),
                    bucket_size=bucket_size,
                )
                hp_manager.time_hp_last_granted = time.time()
                hp_manager.hyperparameter = hp
                return json.dumps(
                    {
                        "recommended_hyperparameters": hp.dict(),
                    }
                )

        @app.route("/api/v1/report_metrics", methods=["POST"])
        def report_metrics():
            req: dict = request.get_json(force=True)
            model_name: str = req["model_name"]
            rank: int = req["rank"]
            train_iter: int = req["train_iter"]
            speed: float = req["speed"]
            hyperparameters = req["hyperparameters"]

            if model_name not in self.model_dict:
                return "Service not ready for report_metrics!", 405

            hp_manager = self.model_dict[model_name]

            # Only consider the rank of the first report metrics now.
            with hp_manager.lock:
                (last_report_train_iter, _, _) = hp_manager.inner.tail_record()
                if train_iter <= last_report_train_iter:
                    return json.dumps({})

                logging.debug(
                    "rank={}, train_iter={}, speed={}, "
                    "hyperparameters={}".format(
                        rank,
                        train_iter,
                        speed,
                        hyperparameters,
                    )
                )
                hp_manager.inner.report_metrics(
                    train_iter=train_iter,
                    hyperparameter=BaguaHyperparameter().update(hyperparameters),
                    system_efficiency_score=speed,
                )

            return json.dumps({})

        @app.route("/api/v1/ask_hyperparameters", methods=["POST"])
        def ask_hyperparameters():
            """
            report_metrics must be called before ask_hyperparameters
            """
            req: dict = request.get_json(force=True)
            rank: int = req["rank"]
            model_name: str = req["model_name"]
            train_iter: int = req["train_iter"]

            if model_name not in self.model_dict:
                return "Service not ready for report_metrics!", 405

            hp_manager = self.model_dict[model_name]

            tensor_partial_order = {}
            with self.tensor_partial_order_lock:
                tensor_partial_order = copy.deepcopy(self.tensor_partial_order)

            logging.debug("tensor_partial_order={}".format(tensor_partial_order))

            with hp_manager.lock:
                # Autotune conditions:
                # 1. autotune_level >= 1.
                # 2. The bagua process is not in the process of hyperparameter update. (self.check_board.count(self.check_board[0])
                #   == len(self.check_board))
                # 3. Only execute autotune at most once in an iteration. (self.check_board[rank] < train_iter)
                check_board = hp_manager.check_board
                if (
                    self.autotune_level >= 1
                    and check_board.count(check_board[0])  # noqa: W503
                    == len(check_board)  # noqa: W503
                    and check_board[rank] < train_iter  # noqa: W503
                ):
                    self.autotune(hp_manager, rank, train_iter, tensor_partial_order)

                check_board[rank] = train_iter

                return json.dumps(
                    {
                        "recommended_hyperparameters": hp_manager.hyperparameter.dict(),
                        "is_autotune_completed": hp_manager.sampling_count
                        > self.max_samples,  # noqa: W503
                    }
                )

        @app.route("/api/v1/report_tensor_execution_order", methods=["POST"])
        def report_tensor_execution_order():
            req: dict = request.get_json(force=True)
            spans: List[BaguaCoreTelemetrySpan] = req["spans"]

            with self.tensor_partial_order_lock:
                spans = sorted(spans, key=lambda span: span["start_time"])
                for span in spans:
                    tensor_name = span["tensor_name"]
                    action = span["action"]

                    if (tensor_name, action) in self.trace_info_dict:
                        continue

                    self.trace_info_dict[(tensor_name, action)] = True
                    if tensor_name not in self.tensor_partial_order:
                        self.tensor_partial_order[tensor_name] = len(
                            self.tensor_partial_order
                        )

            return json.dumps({})

        # set secret-key
        app.config.update(SECRET_KEY=os.urandom(24))

        return app


class AutotuneClient:
    def __init__(
        self,
        service_addr: str,
        service_port: int,
        proxies={
            "http": None,
            "https": None,
        },
    ):
        self.autotune_service_addr = "{}:{}".format(service_addr, service_port)
        self.session = requests.Session()
        self.proxies = proxies

    def report_metrics(
        self,
        model_name: str,
        rank: int,
        train_iter: int,
        hyperparameters: dict,
        speed: float,
    ) -> requests.Response:
        rsp = self.session.post(
            "http://{}/api/v1/report_metrics".format(self.autotune_service_addr),
            json={
                "model_name": model_name,
                "rank": rank,
                "train_iter": train_iter,
                "hyperparameters": hyperparameters,
                "speed": speed,
            },
            proxies=self.proxies,
        )
        return rsp

    def register_tensors(
        self,
        model_name: str,
        tensor_list: List[TensorDeclaration],
        whether_to_bucket: bool = True,
    ) -> requests.Response:
        rsp = self.session.post(
            "http://{}/api/v1/register_tensors".format(self.autotune_service_addr),
            json={
                "model_name": model_name,
                "tensor_list": tensor_list,
                "whether_to_bucket": whether_to_bucket,
            },
            proxies=self.proxies,
        )
        return rsp

    def ask_hyperparameters(
        self,
        model_name: str,
        rank: int,
        train_iter: int,
    ) -> requests.Response:
        rsp = self.session.post(
            "http://{}/api/v1/ask_hyperparameters".format(self.autotune_service_addr),
            json={
                "model_name": model_name,
                "rank": rank,
                "train_iter": train_iter,
            },
            proxies=self.proxies,
        )
        return rsp

    def report_tensor_execution_order(
        self,
        spans: List[BaguaCoreTelemetrySpan],
    ) -> requests.Response:
        rsp = self.session.post(
            "http://{}/api/v1/report_tensor_execution_order".format(
                self.autotune_service_addr
            ),
            json={
                "spans": spans,
            },
            proxies=self.proxies,
        )
        return rsp


if __name__ == "__main__":
    import argparse
    from flask import Flask

    parser = argparse.ArgumentParser()
    parser.add_argument("--nprocs", type=int, default=8)
    parser.add_argument("--port", type=int, default=8123)

    args = parser.parse_args()

    autotune_service = AutotuneService(args.nproc)
    app = Flask(__name__)
    app = autotune_service.setup_app(app)

    server = multiprocessing.Process(
        target=app.run,
        kwargs={
            "host": "0.0.0.0",
            "port": args.port,
        },
    )
    server.daemon = True
    server.start()
    server.join()
