import copy
import csv
import requests
import os
import time
import threading
import tempfile
import json
import logging
import collections
import math
import multiprocessing
from flask import Flask, request
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


def record_autotune_log(
    autotune_logfile_path: str,
    autotune_hp: dict,
    train_iter: int,
    score: float
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


class HyperparameterManager:
    RECORD_MAX_NUM = 1000

    def __init__(
        self,
        is_output_autotune_log: bool,
    ) -> None:
        self.record_deque = collections.deque(
            [
                (
                    -1,
                    BaguaHyperparameter(),
                    float("-inf"),
                )
            ]
        )
        if is_output_autotune_log:
            tmpfile = tempfile.NamedTemporaryFile(
                prefix="bagua_autotune_", mode="w", suffix=".log", delete=False
            )
            self.autotune_logfile_path = tmpfile.name
            tmpfile.close()
        else:
            self.autotune_logfile_path = None

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
        )

    def tail_record(self) -> Tuple[int, BaguaHyperparameter, float]:
        return self.record_deque[-1]

    def best_hyperparameter(self) -> BaguaHyperparameter:
        return sorted(
            [(score, hp) for (_, hp, score) in self.record_deque],
            key=lambda pair: pair[0],
        )[-1][1]

    def report_metrics(
        self,
        train_iter: int,
        hyperparameter: BaguaHyperparameter,
        system_efficiency_score: float,
    ) -> None:
        while len(self.record_deque) > self.RECORD_MAX_NUM:
            self.record_deque.pop()
        self.record_deque.append(
            (
                train_iter,
                hyperparameter,
                system_efficiency_score,
            )
        )

    def ask_hyperparmeter(
        self,
        train_iter: int,
    ) -> BaguaHyperparameter:
        (_, hp, system_efficiency_score) = self.tail_record()
        optimizer_params = {
            "bucket_size_2p": int(math.log(hp.bucket_size, 2)),
            "is_hierarchical_reduce": hp.is_hierarchical_reduce,
        }
        self.bayesian_optimizer.tell(optimizer_params, system_efficiency_score)
        recommend_param = self.bayesian_optimizer.ask()
        recommend_bucket_size = 2 ** recommend_param["bucket_size_2p"]

        if self.autotune_logfile_path:
            record_autotune_log(
                self.autotune_logfile_path,
                optimizer_params,
                train_iter,
                system_efficiency_score,
            )
        tensor_list = [
            tensor_declar for bucket in hp.buckets for tensor_declar in bucket
        ]
        recommend_buckets = split_bucket_by_bucket_size(
            tensor_list,
            recommend_bucket_size,
        )

        recommend_hp = BaguaHyperparameter(
            buckets=recommend_buckets,
            bucket_size=recommend_bucket_size,
            is_hierarchical_reduce=bool(recommend_param["is_hierarchical_reduce"]),
        )

        return recommend_hp


class AutotuneServiceHyperparameterManager:
    def __init__(self, world_size: int, is_output_autotune_log: bool) -> None:
        self.inner = HyperparameterManager(is_output_autotune_log)
        self.warmup_pass_count = 0
        self.sampling_count = 0
        self.lock = threading.Lock()
        self.check_board = [-1] * world_size
        self.time_hp_last_granted = time.time()
        self.hyperparameter = BaguaHyperparameter()


class AutotuneService:
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
        self.model_dict: Dict[str, AutotuneServiceHyperparameterManager] = {}
        self.model_dict_mutex = threading.Lock()

    def autotune(
        self,
        hp_manager: AutotuneServiceHyperparameterManager,
        rank: int,
        train_iter: int,
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
            sampling_time -= self.warmup_time_s

        if sampling_time < self.sampling_confidence_time_s:
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
        recommended_bagua_hp = hp_manager.inner.ask_hyperparmeter(train_iter)
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
                    self.model_dict[model_name] = AutotuneServiceHyperparameterManager(
                        world_size=self.world_size,
                        is_output_autotune_log=self.is_output_autotune_log,
                    )

            hp_manager = self.model_dict[model_name]
            bucket_size = self.default_bucket_size
            if whether_to_bucket is False:
                bucket_size = 10 * 1024 ** 5

            with hp_manager.lock:
                hp = BaguaHyperparameter(
                    buckets=split_bucket_by_bucket_size(
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

            with hp_manager.lock:
                # Autotune conditions:
                # 1. autotune_level >= 1.
                # 2. The bagua process is not in the process of hyperparameter update. (self.check_board.count(self.check_board[0])
                #   == len(self.check_board))
                # 3. Only execute autotune at most once in an iteration. (self.check_board[rank] < train_iter)
                check_board = hp_manager.check_board
                if (
                    self.autotune_level >= 1
                    and check_board.count(check_board[0]) == len(check_board)  # noqa: W503
                    and check_board[rank] < train_iter  # noqa: W503
                ):
                    self.autotune(hp_manager, rank, train_iter)

                check_board[rank] = train_iter

                return json.dumps(
                    {
                        "recommended_hyperparameters": hp_manager.hyperparameter.dict(),
                        "is_autotune_completed": hp_manager.sampling_count
                        > self.max_samples,  # noqa: W503
                    }
                )

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
