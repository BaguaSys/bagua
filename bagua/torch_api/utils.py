from collections import OrderedDict
import torch.distributed as dist
import torch
import math
import time
import logging
import numpy as np
from typing import Tuple, List

LOGGER = logging.getLogger(__name__)

flatten = torch._utils._flatten_dense_tensors
unflatten = torch._utils._unflatten_dense_tensors


def apply_flattened_call(bucket, call, extra_args=None):
    coalesced = flatten(bucket)

    if extra_args is not None:
        call(coalesced, *extra_args)
    else:
        call(coalesced)

    if call is dist.all_reduce:
        coalesced /= dist.get_world_size()

    for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
        buf.copy_(synced)


def _group_by_tensor_type(tensors):
    buckets = OrderedDict()
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)
    return buckets


def apply_flattened_call_all(tensors, call):
    """
    Apply call on a list of tensors.
    """

    grouped_tensors = _group_by_tensor_type(tensors)
    for tensors in grouped_tensors.values():
        apply_flattened_call(tensors, call)


def check_contiguous(tensors):
    data_ptr = None
    for t in tensors:
        if data_ptr is not None and t.data_ptr() != data_ptr:
            return False
        data_ptr = t.data_ptr() + t.numel() * t.element_size()
    return True


def get_flattened_tensor(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 0:
        return

    total_size = 0
    for tensor in tensors:
        total_size += tensor.numel()

    flatten_tensor = torch.zeros(
        total_size, dtype=tensors[0].dtype, device=tensors[0].device
    )

    offset = 0
    for tensor in tensors:
        # copy data
        flatten_tensor[offset : offset + tensor.numel()] = tensor.reshape(-1)
        offset += tensor.numel()

    return flatten_tensor


def to_bagua_datatype(datatype):
    if datatype == torch.float32:
        return "f32"
    elif datatype == torch.float16:
        return "f16"
    elif datatype == torch.uint8:
        return "u8"
    elif datatype == torch.long:
        return "i64"
    else:
        raise ValueError(f"unsupported data type {datatype}.")


def average_by_removing_extreme_values(raw_score_list):
    score_list = np.asarray(raw_score_list)

    # weed out warm up data
    score_list = score_list[len(score_list) // 3 :]

    def weed_out_outliers(X):
        mean = np.mean(X)
        std = np.std(X)
        distance_from_mean = abs(X - mean)
        max_deivations = 1
        not_outlier = distance_from_mean < max_deivations * std
        not_outliers = X[not_outlier]

        if len(not_outliers) == 0:
            return X

        return not_outliers

    score_list = weed_out_outliers(score_list)

    # Repeat up to ten times
    for i in range(10):
        if np.std(score_list) < np.mean(score_list):
            break
        score_list = weed_out_outliers(score_list)

    # score = np.mean(score_list) # TODO: @shjwudp check whether these are still needed
    # std = np.std(score_list)

    return np.mean(score_list), np.std(score_list), score_list.tolist()


class StatisticalAverage:
    def __init__(
        self,
        last_update_time: float = time.time(),
        records: List[float] = [],
        record_tail: Tuple[float, float] = (0.0, 0.0),  # [tail_len, tail_val]
    ) -> None:
        """Track and record the average over a period of time.

        Args:
            last_update_time (float, optional): last update time.
                Defaults to time.time().
            records (List[float], optional): statistical average value from
                `last_update_time`, records[i] is the average value from
                last_update_time to last_update_time + 2 ^ i (unit: seconds).
                Defaults to [].
            tail (Tuple[float, float], optional): tail of record, first one
                is tail length (unit: seconds), second one is tail average
                value. Defaults to (0., 0.).
        """
        self.last_update_time: float = last_update_time
        self.records: List[float] = records
        self.record_tail: Tuple[float, float] = record_tail

    def record_seconds(self) -> float:
        return 2.0 ** (len(self.records) - 1) if len(self.records) != 0 else 0.0

    def total_recording_time(self) -> float:
        (tail_seconds, _) = self.record_tail

        return self.record_seconds() + tail_seconds

    def get_records_mean(self, last_n_seconds: float) -> float:
        if last_n_seconds <= 0.0:
            return 0.0

        records_seconds = self.record_seconds()
        (tail_seconds, tail_mean) = self.record_tail

        if len(self.records) == 0:
            return tail_mean

        if last_n_seconds < 1.0:
            return self.records[0]

        if last_n_seconds <= records_seconds:
            floor_id = max(0, math.floor(math.log(last_n_seconds, 2.0)))
            floor_time = 2.0 ** floor_id
            if floor_id + 1 < len(self.records):
                a, b = self.records[floor_id], self.records[floor_id + 1]
                a_l, b_l = floor_time, floor_time * 2.0
                mean = a + (b - a) * (last_n_seconds - a_l) / (b_l - a_l)
            else:
                mean = self.records[floor_id]
        elif last_n_seconds <= records_seconds + tail_seconds:
            a, b = self.records[-1], tail_mean
            a_l, b_l = records_seconds, records_seconds + tail_seconds
            mean = a + (b - a) * (last_n_seconds - a_l) / (b_l - a_l)
        else:
            mean = tail_mean

        return mean

    def record(self, val: float):
        now = time.time()
        time_dist: float = now - self.last_update_time
        new_records: List[float] = []
        new_tail: Tuple[float, float] = (0.0, 0.0)

        for i in range(64):
            coverage_period = 2.0 ** i

            if coverage_period <= time_dist:
                new_records.append(val)
            elif coverage_period <= time_dist + self.total_recording_time():
                record_contribution_percentage = time_dist / coverage_period
                new_val = val * record_contribution_percentage + self.get_records_mean(
                    coverage_period - time_dist
                ) * (1.0 - record_contribution_percentage)

                new_records.append(new_val)

                if coverage_period > time_dist + self.total_recording_time():
                    break
            else:
                new_total_time = time_dist + self.total_recording_time()
                report_contribution_percentage = time_dist / new_total_time

                tail_len = new_total_time - 2.0 ** (len(new_records) - 1)
                tail_val = val * report_contribution_percentage + self.get_records_mean(
                    self.total_recording_time()
                ) * (1.0 - report_contribution_percentage)
                new_tail = (tail_len, tail_val)
                break

        self.last_update_time = now
        self.records = new_records
        self.record_tail = new_tail

    def get(self, last_n_seconds: float) -> float:
        time_dist = time.time() - self.last_update_time
        if last_n_seconds <= time_dist:
            if len(self.records) != 0:
                return self.records[0]
            else:
                (tail_mean, _) = self.record_tail
                return tail_mean

        return self.get_records_mean(last_n_seconds - time_dist)

    def __str__(self) -> str:
        return str(
            {
                "last_update_time": self.last_update_time,
                "records": self.records,
                "record_tail": self.record_tail,
            }
        )
