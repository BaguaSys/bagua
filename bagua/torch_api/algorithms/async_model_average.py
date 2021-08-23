#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm
from typing import List
from bagua.torch_api.tensor import BaguaTensor
import threading
import time
import logging
import os


def check_nccl_proto():
    # TODO: remove nccl proto check
    proto_str = os.environ.get("NCCL_PROTO", "")
    if (
        proto_str == ""
        or ("^" not in proto_str and "LL128" in proto_str)  # noqa: W503
        or ("^" in proto_str and "LL128" not in proto_str)  # noqa: W503
    ):
        logging.warn(
            "`LL128` proto for NCCL backend is not stable for async algorithms. Set `NCCL_PROTO=^LL128` to exclude it."
        )  # TODO; remove this after https://github.com/NVIDIA/nccl/issues/549 gets solved


class AsyncModelAverageAlgorithm(Algorithm):
    def __init__(
        self, peer_selection_mode: str = "all", sync_interval_ms: int = 500,
    ):
        """
        Create an instance of the
        `AsyncModelAverage <https://bagua-tutorials.kwai-seattle.com/algorithms/async-model-average.html>`_
        algorithm.

        The asynchronous implementation is experimental, and imposes some restrictions.
        With such asynchronous algorithm, the number of iterations on each worker are different. Therefore
        the current implementation assumes that the dataset is an endless stream, and all workers continuously
        synchronize between each other.

        Users should call :func:`abort` to manually stop the algorithm's continuous synchronization process.

        Args:
            peer_selection_mode (str): The way how workers communicate with each other. Currently "all" is supported.
                "all" means all workers' weights are synchronized during each communication.
            sync_interval_ms (int): Number of milliseconds between model synchronizations.
        """

        self.peer_selection_mode = peer_selection_mode
        self.sync_interval_ms = sync_interval_ms
        self.stop_event = threading.Event()
        check_nccl_proto()

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        parameters = bagua_module.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name, bagua_module.bagua_module_name)
            for name, param in parameters.__reversed__()
        ]
        return self.tensors

    def init_forward_pre_hook(self, bagua_module: BaguaModule):
        def hook(input):
            if not hasattr(self, "worker"):
                self.worker = threading.Thread(
                    target=self.run_async_loop, args=[bagua_module]
                )
                self.worker.start()

        return hook

    def init_backward_hook(self, bagua_module: BaguaModule):
        def hook(parameter_name, parameter):
            pass

        return hook

    def init_post_backward_hook(self, bagua_module: BaguaModule):
        def hook():
            pass

        return hook

    def init_operations(
        self, bagua_module: BaguaModule, bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        bucket.append_asynchronous_model_average_op(
            peer_selection_mode=self.peer_selection_mode,
        )

    def abort(self, bagua_module: BaguaModule, grace_period_seconds=5):
        """
        Gracefully stop all workers.

        Args:
            bagua_module: A PyTorch module initialized by ``with_bagua(...)`` method.
            grace_period_seconds: Number of seconds a worker will wait before aborting its unfinished communication operations.
        """
        assert (
            self.worker.is_alive()  # pytype: disable=attribute-error
        ), "cannot abort since the asynchronous communication thread is not started"
        self.stop_event.set()
        time.sleep(grace_period_seconds)
        bagua_module._bagua_backend.global_communicator.abort()

        self.worker.join()  # pytype: disable=attribute-error

    def run_async_loop(self, bagua_module: BaguaModule):
        while not self.stop_event.is_set():
            if bagua_module.training:
                for bucket in bagua_module.bagua_buckets:
                    for tensor in bucket.tensors:
                        tensor.bagua_mark_communication_ready_without_synchronization()

                bagua_module._bagua_backend.wait_pending_comm_ops()

            time.sleep(self.sync_interval_ms / 1000)
