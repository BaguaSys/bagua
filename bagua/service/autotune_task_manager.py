import collections
import tempfile
import copy
import math
import logging
import csv
from typing import Tuple, List, Dict

from .bayesian_optimizer import (
    IntParam,
    BoolParam,
    BayesianOptimizer,
)
from bagua.bagua_define import (
    TensorDtype,
    TensorDeclaration,
    BaguaHyperparameter,
)


class AutotuneTaskManager:
    RECORD_MAX_NUM = 1000

    def __init__(
        self,
        task_name,
        need_to_log: bool,
    ) -> None:
        self.task_name = task_name
        self.record_deque = collections.deque(
            [
                (
                    -1,
                    BaguaHyperparameter(),
                    float("-inf"),
                )
            ]
        )
        if need_to_log:
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

    @staticmethod
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
            cols = collections.OrderedDict(cols)
            logging.info("cols={}".format(cols))
            csv_writer.writerow(cols)

    @staticmethod
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
        tensor_partial_order: Dict[str, int] = {},  # tensor_name -> rank
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
            AutotuneTaskManager.record_autotune_log(
                self.autotune_logfile_path,
                optimizer_params,
                train_iter,
                system_efficiency_score,
            )
        tensor_list = [
            tensor_declar for bucket in hp.buckets for tensor_declar in bucket
        ]
        tensor_list = sorted(
            tensor_list, key=lambda td: tensor_partial_order.get(td["name"], -1)
        )

        recommend_buckets = AutotuneTaskManager.split_bucket_by_bucket_size(
            tensor_list,
            recommend_bucket_size,
        )

        recommend_hp = BaguaHyperparameter(
            buckets=recommend_buckets,
            bucket_size=recommend_bucket_size,
            is_hierarchical_reduce=bool(recommend_param["is_hierarchical_reduce"]),
        )

        return recommend_hp
