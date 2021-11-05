from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.communication import BaguaProcessGroup
from typing import List
import torch


class Algorithm:
    """
    This is the base class that all Bagua algorithms inherit.
    """

    def reify(self, process_group: BaguaProcessGroup):
        """
        Create an algorithm instance.

        Args:
            process_group: The process group to work on.
        """
        pass


class AlgorithmImpl:
    """
    This is the base class that all Bagua algorithm implementations inherit.

    It provides methods that can be override to implement different kinds of
    distributed algorithms.

    Args:
        process_group: The process group to work on.
    """

    def __init__(self, process_group: BaguaProcessGroup):
        self.process_group = process_group

    def need_reset(self) -> bool:
        """
        Returns:
             ``True`` if all initialization methods of the current algorithms should be called again. \
             This is useful for algorithms that have multiple stages where each stage needs different initializations.
        """
        return False

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        """
        Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, return Bagua tensors to be used in Bagua for later
        operations.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.

        Returns:
            A list of Bagua tensors for communication.
        """

        parameters = bagua_ddp.bagua_build_params()
        tensors = []
        for name, param in parameters.__reversed__():
            param = param.bagua_ensure_grad().ensure_bagua_tensor(
                name,
                bagua_ddp.bagua_module_name,
                getter_closure=lambda param: param.grad,
                setter_closure=lambda param, t: setattr(param, "grad", t),
            )
            tensors.append(param)

        self._communication_tensor_names = set(name for name, _ in parameters)
        assert len(self._communication_tensor_names) == len(
            tensors
        ), "tensor names should be unique"
        return tensors

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        """
        Given the bucketing suggestion from Bagua, return the actual Bagua buckets.
        The default implementation follows the suggestion to do the bucketing.

        Args:
            tensors: Bagua tensors grouped in different
                lists, representing Bagua's suggestion on how to bucketing the
                tensors.
            do_flatten: Whether to flatten the Bagua buckets.

        Returns:
            A list of Bagua buckets.
        """
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(
                bucket, flatten=do_flatten, name=str(idx)
            )  # TODO: check duplicated names
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        """Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, return a hook function that will be executed before the
        forward process.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.

        Returns:
            A function that takes the model's input.
        """

        def hook(input):
            pass

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        """Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, return a hook function that will be executed on every
        parameter's gradient computation completion.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.

        Returns:
            A function that takes the name of a parameter (as in ``torch.nn.Module.named_parameters``) and the parameter itself.
        """

        def hook(parameter_name, parameter):
            if parameter_name in self._communication_tensor_names:
                assert (
                    parameter.bagua_backend_tensor().data_ptr()
                    == parameter.grad.data_ptr()
                ), "bagua backend tensor data_ptr should match parameter grad"
                parameter.bagua_mark_communication_ready()

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        """Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, return a hook function that will be executed when the
        backward pass is done.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.

        Returns:
            A function that takes no argument.
        """

        def hook():
            bagua_ddp._bagua_backend.wait_pending_comm_ops()

        return hook

    def init_post_optimizer_step_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        """Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, return a hook function that will be executed when the
        ``optimizer.step()`` is done.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.

        Returns:
            A function that gets called after an optimizer's ``step()`` method is called. The function takes the optimizer as its argument.
        """

        def hook(optimizer: torch.optim.Optimizer):
            pass

        return hook

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        """Given a :class:`~bagua.torch_api.data_parallel.BaguaDistributedDataParallel`, and a :class:`~bagua.torch_api.bucket.BaguaBucket`,
        register operations to be executed on the bucket.

        Args:
            bagua_ddp: :class:`bagua.torch_api.data_parallel.BaguaDistributedDataParallel`.
            bucket: A single bucket to register operations.
        """
