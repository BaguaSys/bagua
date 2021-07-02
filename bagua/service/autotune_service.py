import copy
import csv
import requests
import os
import time
import threading
import json
import logging
import math
import multiprocessing
from flask import Flask, request, session
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import prometheus_client
from prometheus_client import make_wsgi_app
from .autotune import BayesianOptimizer, IntParam, BoolParam
from bagua.bagua_define import (
    TensorDtype,
    TensorDeclaration,
    BaguaHyperparameter,
)
import numpy as np
from typing import Dict, List, Tuple
from collections import OrderedDict


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


# TODO: There is only fusion logic, should the logic of splitting tensor be added?
def split_bucket_by_bucket_size(
    tensor_list: List[TensorDeclaration],
    bucket_size: int,
    param_group_info: Dict[str, int] = {},
):
    dtype_unit_size = {
        TensorDtype.F32.value: 4,
        TensorDtype.F16.value: 2,
        TensorDtype.U8.value: 1,
    }

    buckets = []
    tmp_bucket = []
    tmp_bucket_size = 0
    for (dtype, unit_size) in sorted(dtype_unit_size.items()):
        for tensor_declar in [x for x in tensor_list if x["dtype"] == dtype]:
            tensor_size = tensor_declar["num_elements"] * unit_size
            tmp_bucket_size += tensor_size
            tmp_bucket.append(tensor_declar)
            if tmp_bucket_size >= bucket_size:
                buckets.append(tmp_bucket)
                tmp_bucket = []
                tmp_bucket_size = 0

    if len(tmp_bucket) != 0:
        buckets.append(tmp_bucket)

    # group params in bucket
    for i, _ in enumerate(buckets):
        buckets[i] = sorted(
            buckets[i], key=lambda p: param_group_info.get(p["name"], -1)
        )

    return buckets


try:
    rank0_hyperparameters = prometheus_client.Info(
        "rank0_hyperparameters", "The newly reported rank0 hyperparameter."
    )
    rank0_train_iter = prometheus_client.Gauge(
        "rank0_train_iter", "The iteration when parameters reporting."
    )
    rank0_speed = prometheus_client.Gauge(
        "rank0_speed",
        "Data used for hyperparameters evaluation.",
    )
except ValueError:
    pass


def record_autotune_log(
    autotune_logfile_path: str, autotune_hp: dict, train_iter: int, score: float
):
    with open(autotune_logfile_path, "a") as autotune_log:
        csv_writer = csv.DictWriter(
            autotune_log,
            fieldnames=sorted(["train_iter", "score"] + list(autotune_hp.keys())),
        )
        first_line = open(autotune_logfile_path).readline()
        if not first_line:
            csv_writer.writeheader()

        cols = copy.deepcopy(autotune_hp)
        cols.update(
            {
                "train_iter": train_iter,
                "score": score,
            }
        )
        cols = OrderedDict(cols)
        logging.info("cols={}".format(cols))
        csv_writer.writerow(cols)


class AutotuneService:
    def __init__(
        self,
        world_size,
        autotune_level=0,
        max_samples=60,
        sampling_confidence_time_s=5,
        warmup_time_s=30,
        autotune_logfile_path="/tmp/bagua_autotune.log",
        default_bucket_size=10 * 1024 ** 2,
    ):
        self.autotune_level = autotune_level
        self.world_size = world_size
        self.max_samples = max_samples
        self.sampling_confidence_time_s = sampling_confidence_time_s
        self.warmup_time_s = warmup_time_s
        self.warmup_flag = False
        self.is_initialized = False
        self.autotune_logfile_path = autotune_logfile_path
        if self.autotune_level >= 1:
            try:
                os.remove(self.autotune_logfile_path)
            except OSError:
                pass

        self.bayesian_optimizer = BayesianOptimizer(
            {
                "bucket_size_2p": IntParam(  # bucket_size = 2 ^ bucket_size_2p
                    val=13,
                    space_dimension=(  # 1KB ~ 2GB
                        10,
                        31,
                    ),
                ),
                "is_hierarchical_reduce": BoolParam(False),
            },
            n_initial_points=10,
        )
        self.hyperparameters_and_score_list: list = []
        self.sampling_counter: int = 0  # 采样计数器
        self.check_board = [-1] * self.world_size
        self.last_time_the_hyperparameters_was_granted = (
            time.time()
        )  # 记录上次超参授予bagua进程的时刻，用于衡量采样时间

        self.ask_hyperparameters_mutex = threading.Lock()
        self.recommended_hyperparameters: BaguaHyperparameter = BaguaHyperparameter()
        self.recommended_from_iter = 0
        self.default_bucket_size: int = default_bucket_size
        self.forerunner_bucket_size: int = -1

        # metrics to autotune
        self.metrics_mutex = threading.Lock()
        rank0_hyperparameters.info(self.recommended_hyperparameters.dict())
        rank0_train_iter.set(-1)
        rank0_speed.set(-1)

    def tell_and_ask_bayesian_recommended_hyperparameters(
        self, bagua_hp, score, train_iter, bucket_size_2p
    ) -> Tuple[BaguaHyperparameter, Dict]:
        autotune_hp = {
            "bucket_size_2p": bucket_size_2p,
            "is_hierarchical_reduce": bagua_hp.is_hierarchical_reduce,
        }
        self.bayesian_optimizer.tell(autotune_hp, score)
        recommended_autotune_hp = self.bayesian_optimizer.ask()
        recommended_bucket_size = 2 ** recommended_autotune_hp["bucket_size_2p"]

        logging.info(
            "autotune_hp={}, train_iter={}, score={}".format(
                autotune_hp, train_iter, score
            )
        )
        record_autotune_log(self.autotune_logfile_path, autotune_hp, train_iter, score)

        tensor_list = [
            tensor_declar for bucket in bagua_hp.buckets for tensor_declar in bucket
        ]
        recommended_buckets = split_bucket_by_bucket_size(
            tensor_list,
            recommended_bucket_size,
        )

        recommended_bagua_hp = BaguaHyperparameter(
            buckets=recommended_buckets,
            is_hierarchical_reduce=bool(
                recommended_autotune_hp["is_hierarchical_reduce"]
            ),
        )

        return recommended_bagua_hp, recommended_autotune_hp

    def setup_app(self, app):
        @app.route("/api/v1/bagua_backend_metrics", methods=["POST"])
        def report_bagua_backend_metrics():
            req: dict = request.get_json(force=True)  # TODO: @shjwudp
            tensor_ready_order: list = req["tensor_ready_order"]
            communication_time_ms: float = req["communication_time_ms"]
            hyperparameters: dict = req["hyperparameters"]

        @app.route("/api/v1/register_tensors", methods=["POST"])
        def register_tensors():
            req: dict = request.get_json(force=True)
            tensor_list: list[TensorDeclaration] = req["tensor_list"]
            whether_to_bucket: bool = req["whether_to_bucket"]

            session["bucket_size"] = self.default_bucket_size
            if whether_to_bucket is False:
                session["bucket_size"] = (
                    10 * 1024 ** 5
                )  # if you don't divide buckets, set big bucket as 10PB.

            with self.ask_hyperparameters_mutex:
                default_hyperparameters = BaguaHyperparameter(
                    buckets=split_bucket_by_bucket_size(
                        tensor_list,
                        session["bucket_size"],
                    ),
                )
                if self.is_initialized:
                    return json.dumps(
                        {
                            "message": "Autotune service has been initialized, and the current operation does not take effect.",
                            "recommended_hyperparameters": default_hyperparameters.dict(),
                        },
                    )

                self.recommended_hyperparameters = default_hyperparameters
                self.last_time_the_hyperparameters_was_granted = time.time()

                logging.info(
                    "default_hyperparameters={}".format(default_hyperparameters.dict())
                )
                self.is_initialized = True
                return json.dumps(
                    {
                        "recommended_hyperparameters": default_hyperparameters.dict(),
                    }
                )

        @app.route("/api/v1/report_metrics", methods=["POST"])
        def report_metrics():
            req: dict = request.get_json(force=True)
            rank: int = req["rank"]
            train_iter: int = req["train_iter"]
            speed: float = req["speed"]
            hyperparameters = req["hyperparameters"]

            if not self.is_initialized:
                return "Service not ready for ask_hyperparameters!", 405

            # Only consider the rank of the first report metrics now.
            with self.metrics_mutex:
                if train_iter <= rank0_train_iter.collect()[-1].samples[-1].value:
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
                rank0_hyperparameters.info(hyperparameters)
                rank0_train_iter.set(train_iter)
                rank0_speed.set(speed)

            return json.dumps({})

        @app.route("/api/v1/checkboard", methods=["GET"])
        def request_checkboard():
            return json.dumps(
                {
                    "check_board": self.check_board,
                }
            )

        @app.route("/api/v1/ask_hyperparameters", methods=["POST"])
        def ask_hyperparameters():
            """
            report_metrics must be called before ask_hyperparameters
            """
            req: dict = request.get_json(force=True)
            rank: int = req["rank"]
            train_iter: int = req["train_iter"]

            if not self.is_initialized:
                return "Service not ready for ask_hyperparameters!", 405

            # # The bucket parameters requires at least one metrics report
            # while len(rank0_hyperparameters.collect()) == 0:
            #     time.sleep(0)

            def autotune():
                if self.sampling_counter > self.max_samples:
                    return

                with self.metrics_mutex:
                    recommended_train_iter = int(
                        rank0_train_iter.collect()[-1].samples[-1].value
                    )
                    speed = rank0_speed.collect()[-1].samples[-1].value
                    hyperparameters = (
                        rank0_hyperparameters.collect()[-1].samples[-1].labels
                    )
                    self.hyperparameters_and_score_list.append(
                        [
                            BaguaHyperparameter(**hyperparameters),
                            speed,
                        ]
                    )
                    hyperparameters_and_score_list = copy.deepcopy(
                        self.hyperparameters_and_score_list
                    )

                    logging.info(
                        "recommended_train_iter={}, hyperparameters={}, speed={}".format(
                            recommended_train_iter,
                            hyperparameters,
                            speed,
                        )
                    )

                sampling_time = (
                    time.time() - self.last_time_the_hyperparameters_was_granted
                )

                # warmup pass
                if self.warmup_flag:
                    # At last pass one time
                    if (
                        sampling_time < self.warmup_time_s
                        or self.warmup_pass_count == 0  # noqa: W503
                    ):
                        logging.info(
                            "warmup pass, time.time={}, last={}, "
                            "warmup_time_s={}".format(
                                time.time(),
                                self.last_time_the_hyperparameters_was_granted,
                                self.warmup_time_s,
                            )
                        )
                        self.warmup_pass_count += 1
                        return
                    self.last_time_the_hyperparameters_was_granted = time.time()
                    self.warmup_flag = False

                # Skip if the sampling time is insufficient
                if sampling_time < self.sampling_confidence_time_s:
                    logging.debug(
                        "Insufficient sampling time, time={}, last={}, "
                        "sampling_confidence_time_s={}".format(
                            time.time(),
                            self.last_time_the_hyperparameters_was_granted,
                            self.sampling_confidence_time_s,
                        )
                    )
                    return

                logging.info(
                    "rank={}, train_iter={}, sampling_counter={}, "
                    "max_samples={}".format(
                        rank, train_iter, self.sampling_counter, self.max_samples
                    )
                )
                bagua_hp = BaguaHyperparameter(**hyperparameters)
                score = speed
                bucket_size_2p = int(math.log(session["bucket_size"], 2))
                (
                    recommended_bagua_hp,
                    recommended_autotune_hp,
                ) = self.tell_and_ask_bayesian_recommended_hyperparameters(
                    bagua_hp,
                    score,
                    recommended_train_iter,
                    bucket_size_2p,
                )

                if self.sampling_counter < self.max_samples:
                    self.recommended_hyperparameters = recommended_bagua_hp
                    self.recommended_from_iter = recommended_train_iter
                    self.forerunner_bucket_size = int(
                        2 ** recommended_autotune_hp["bucket_size_2p"]
                    )
                else:
                    # get best hyperparameters
                    sorted_score_hp = sorted(
                        [
                            (s, copy.deepcopy(p))
                            for p, s in hyperparameters_and_score_list
                        ],
                        key=lambda score_hp: score_hp[0],
                        reverse=True,
                    )
                    logging.info(
                        "sorted_score_hp={}".format(
                            [(s, p.dict()) for s, p in sorted_score_hp]
                        )
                    )
                    self.recommended_hyperparameters = sorted_score_hp[0][1]
                    self.recommended_from_iter = recommended_train_iter

                # The hyperparameters has been granted, update the time
                # NOTE: The accuracy depends on the client calling ask_hyperparameters at the right time and success
                self.last_time_the_hyperparameters_was_granted = time.time()
                self.sampling_counter += 1

            with self.ask_hyperparameters_mutex:
                # Autotune conditions:
                # 1. autotune_level >= 1.
                # 2. The bagua process is not in the process of hyperparameter update. (self.check_board.count(self.check_board[0])
                #   == len(self.check_board))
                # 3. Only execute autotune at most once in an iteration. (self.check_board[rank] < train_iter)
                if (
                    self.autotune_level >= 1
                    and self.check_board.count(self.check_board[0])
                    == len(self.check_board)
                    and self.check_board[rank] < train_iter
                ):
                    self.forerunner_bucket_size: int = session["bucket_size"]
                    autotune()

                session["bucket_size"] = self.forerunner_bucket_size
                logging.debug(
                    "bucket_size={}, tarin_iter={}, rank={}".format(
                        session["bucket_size"], train_iter, rank
                    )
                )
                self.check_board[rank] = train_iter

                return json.dumps(
                    {
                        "recommended_hyperparameters": self.recommended_hyperparameters.dict(),
                        "recommended_from_iter": self.recommended_from_iter,
                        # fmt: off
                        "is_autotune_completed":
                            self.sampling_counter > self.max_samples,
                    },
                )

        @app.route("/api/v1/reset", methods=["POST"])
        def reset():
            with self.ask_hyperparameters_mutex:
                if self.is_initialized:
                    self.__init__(
                        world_size=self.world_size,
                        autotune_level=self.autotune_level,
                        max_samples=self.max_samples,
                        sampling_confidence_time_s=self.sampling_confidence_time_s,
                    )

            return json.dumps({})

        # Add prometheus wsgi middleware to route /metrics requests
        app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})

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
        rank: int,
        unix_timestamp: float,
        train_iter: int,
        hyperparameters: dict,
        speed: float,
    ) -> requests.Response:
        rsp = self.session.post(
            "http://{}/api/v1/report_metrics".format(self.autotune_service_addr),
            json={
                "rank": rank,
                "unix_timestamp": unix_timestamp,
                "train_iter": train_iter,
                "hyperparameters": hyperparameters,
                "speed": speed,
            },
            proxies=self.proxies,
        )
        return rsp

    def register_tensors(
        self,
        tensor_list: List[TensorDeclaration],
        whether_to_bucket: bool = True,
    ) -> requests.Response:
        rsp = self.session.post(
            "http://{}/api/v1/register_tensors".format(self.autotune_service_addr),
            json={
                "tensor_list": tensor_list,
                "whether_to_bucket": whether_to_bucket,
            },
            proxies=self.proxies,
        )
        return rsp

    def ask_hyperparameters(
        self,
        rank: int,
        train_iter: int,
    ) -> requests.Response:
        rsp = self.session.post(
            "http://{}/api/v1/ask_hyperparameters".format(self.autotune_service_addr),
            json={
                "rank": rank,
                "train_iter": train_iter,
            },
            proxies=self.proxies,
        )
        return rsp


if __name__ == "__main__":
    autotune_service = AutotuneService(10)
    app = Flask(__name__)
    app = autotune_service.setup_app(app)

    server = multiprocessing.Process(
        target=app.run,
        kwargs={
            "host": "0.0.0.0",
            "port": 8123,
        },
    )
    server.daemon = True
    server.start()
    server.join()
