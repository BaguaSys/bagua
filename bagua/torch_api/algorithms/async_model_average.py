#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.distributed import BaguaModule
from bagua.torch_api.algorithms import Algorithm
from typing import List
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.env import get_rank
from enum import IntEnum
import threading
import time
import os
import torch
import atexit
import logging

__all__ = ["AsyncModelAverageAlgorithm"]


def check_nccl_proto():
    # TODO: remove nccl proto check
    proto_str = os.environ.get("NCCL_PROTO", "")
    if (
        proto_str == ""
        or ("^" not in proto_str and "LL128" in proto_str)  # noqa: W503
        or ("^" in proto_str and "LL128" not in proto_str)  # noqa: W503
    ):
        print(
            "Warning: `LL128` proto for NCCL backend is not stable for async algorithms. Set `NCCL_PROTO=^LL128` to exclude it."
        )  # TODO; remove this after https://github.com/NVIDIA/nccl/issues/549 gets solved


class _AsyncInternalState(IntEnum):
    RESUME = 0
    ABORT = 1
    END = 2


class AsyncModelAverageAlgorithm(Algorithm):
    def __init__(
        self,
        peer_selection_mode: str = "all",
        sync_interval_ms: int = 500,
        warmup_steps: int = 0,
    ):
        """
        Create an instance of the
        `AsyncModelAverage <https://bagua-tutorials.kwai-seattle.com/algorithms/async-model-average.html>`_
        algorithm.

        The asynchronous implementation is experimental, and imposes some restrictions.
        With such asynchronous algorithm, the number of iterations on each worker are different. Therefore
        the current implementation assumes that the dataset is an endless stream, and all workers continuously
        synchronize between each other.

        Users should call :meth:`abort` to manually stop the algorithm's continuous synchronization process.

        Args:
            peer_selection_mode (str): The way how workers communicate with each other. Currently ``"all"`` is supported.
                ``"all"`` means all workers' weights are synchronized during each communication.
            sync_interval_ms (int): Number of milliseconds between model synchronizations.
            warmup_steps (int): Number of steps to warm up by gradient allreduce at the beginning of training.
        """

        self.peer_selection_mode = peer_selection_mode
        self.sync_interval_ms = sync_interval_ms
        self.step_id = 0
        self.warmup_steps = warmup_steps

        self.cuda_event = torch.cuda.Event()

        self.abort_event = threading.Event()
        self.end_event = threading.Event()
        self.dummy_tensor = torch.Tensor([0]).byte()
        self.main_group = torch.distributed.new_group(backend="gloo")
        self.thread_group = torch.distributed.new_group(backend="gloo")

        if self.warmup_steps <= 0:
            self.no_bucketing = True

    def init_tensors(self, bagua_module: BaguaModule) -> List[BaguaTensor]:
        parameters = bagua_module.bagua_build_params()
        tensors = []
        for name, param in parameters.__reversed__():
            if self.step_id < self.warmup_steps:
                grad = param.bagua_ensure_grad().ensure_bagua_tensor(
                    name, bagua_module.bagua_module_name
                )
                param._bagua_grad = grad
                tensors.append(grad)
            else:
                p = param.ensure_bagua_tensor(name, bagua_module.bagua_module_name)
                tensors.append(p)

        return tensors

    def init_forward_pre_hook(self, bagua_module: BaguaModule):
        def hook(input):
            if (
                self.step_id > self.warmup_steps
                and self.sync_interval_ms > 0  # noqa: W503
            ):
                torch.cuda.current_stream().record_event(self.cuda_event)
                self.cuda_event.synchronize()
                assert len(bagua_module.bagua_buckets) == 1
                bagua_module.bagua_buckets[0]._async_op.lock_weight()

                if not hasattr(self, "worker"):  # noqa: W503
                    self.worker = threading.Thread(
                        target=self._run_async_loop, args=[bagua_module]
                    )
                    self.worker.start()

        return hook

    def init_backward_hook(self, bagua_module: BaguaModule):
        def hook(parameter_name, parameter):
            if self.step_id <= self.warmup_steps:
                parameter._bagua_grad.bagua_mark_communication_ready()

        return hook

    def init_post_backward_hook(self, bagua_module: BaguaModule):
        def hook():
            if self.step_id <= self.warmup_steps:
                bagua_module._bagua_backend.wait_pending_comm_ops()
            else:
                torch.cuda.current_stream().record_event(self.cuda_event)
                self.cuda_event.synchronize()
                assert len(bagua_module.bagua_buckets) == 1
                bagua_module.bagua_buckets[0]._async_op.unlock_weight()

        return hook

    def need_reset(self):
        self.step_id += 1

        if self.warmup_steps > 0 and self.step_id == self.warmup_steps + 1:
            logging.info(f"Async model average starts from step {self.step_id}")
            self.no_bucketing = True
            return True
        else:
            return False

    def init_operations(
        self,
        bagua_module: BaguaModule,
        bucket: BaguaBucket,
    ):
        bagua_module._bagua_backend.wait_pending_comm_ops()
        bucket.clear_ops()

        if self.step_id < self.warmup_steps:
            bucket.append_centralized_synchronous_op(
                hierarchical=False,
                average=True,
            )
        else:
            async_op = bucket.append_asynchronous_model_average_op(
                peer_selection_mode=self.peer_selection_mode,
            )
            bucket._async_op = async_op

    def _negotiate(self):
        if self.end_event.is_set():
            self.dummy_tensor[0] = _AsyncInternalState.END
        elif self.abort_event.is_set():
            self.dummy_tensor[0] = _AsyncInternalState.ABORT
        else:
            self.dummy_tensor[0] = _AsyncInternalState.RESUME

        torch.distributed.broadcast(self.dummy_tensor, src=0, group=self.thread_group)
        return self.dummy_tensor.item()

    def _run_async_loop(self, bagua_module: BaguaModule):
        comm_step = 0
        while True:
            state = self._negotiate()

            if state == _AsyncInternalState.END:
                break

            if state == _AsyncInternalState.RESUME:
                start_time = time.time()
                for bucket in bagua_module.bagua_buckets:
                    for tensor in bucket.tensors:
                        tensor.bagua_mark_communication_ready_without_synchronization()

                bagua_module._bagua_backend.wait_pending_comm_ops()
                duration = (time.time() - start_time) * 1000

                logging.debug(
                    "Process {} async communication cost {}ms, comm_step={}".format(
                        get_rank(), duration, comm_step
                    )
                )

                comm_step += 1
            time.sleep(self.sync_interval_ms / 1000)

    def abort(self):
        """Abort async communications after training."""

        torch.distributed.barrier(group=self.main_group)
        self.abort_event.set()

    def resume(self):
        """Resume async communications before training."""

        torch.distributed.barrier(group=self.main_group)
        self.abort_event.clear()

    def destroy(self):
        """Cleanup resources at the end of your training."""

        if (
            not hasattr(self, "worker")
            or not self.worker.is_alive()  # pytype: disable=attribute-error # noqa: W503
        ):
            logging.info(
                "Warning: skip abort since the asynchronous communication thread is not started."
            )
            return

        torch.distributed.barrier(group=self.main_group)
        self.end_event.set()
        self.worker.join()  # pytype: disable=attribute-error
