import torch
import math
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
from typing import Optional, Iterator, Callable
from collections import OrderedDict

__all__ = ["LoadBalancingDistributedSampler", "LoadBalancingDistributedBatchSampler"]


class LoadBalancingDistributedSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    This sampler use a :attr:`complexity_fn` to calculate each sample's computational
    complexity and make each batch get similar computation complexity.

    This is useful in scenarios like speech and NLP, where each batch has variable
    length and distributed training suffers from straggler problem.

    The usage is similar to `torch.utils.data.DistributedSampler <https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler>`_,
    where each process loads a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        complexity_fn(Callable): A function whose input is a sample and output is an integer as a
            measure of the computational complexity of the sample.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: 0.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        random_level (float, optional): A float varies from 0 and 1 that controls the extent
            of load balance. 0 means the best load balance, while 1 means the opposite.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the `DataLoader <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`_ iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::
        Define your :attr:`complexity_fn`, which accepts a dataset sample as its input and produces an integer
        as the sample's computational complexity:

        >>> dataset = torch.utils.data.TensorDataset(torch.randn(n, 2), torch.randperm(n))
        >>> complexity_fn = lambda x: x[1]

        Below is the usage of :class:`LoadBalancingDistributedSampler`
        and `DataLoader <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`_:

        >>> sampler = bagua.torch_api.contrib.LoadBalancingDistributedSampler(
        ...     dataset,
        ...     complexity_fn=complexity_fn) if is_distributed else None
        >>> loader = torch.utils.data.DataLoader(dataset,
        ...     shuffle=(sampler is None),
        ...     sampler=sampler)
        >>>
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        complexity_fn: Callable[..., int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        random_level: float = 0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        dataset_len = len(self.dataset)  # type: ignore
        if self.drop_last and dataset_len % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (dataset_len - self.num_replicas)
                / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(dataset_len / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        self.item_complexity_map = dict()
        for item_index in range(dataset_len):
            self.item_complexity_map[item_index] = complexity_fn(
                self.dataset[item_index]
            )

        self.ordered_item_complexity_map = OrderedDict(
            sorted(self.item_complexity_map.items(), key=lambda t: t[1])
        )
        max_complexity = max(self.item_complexity_map.values())
        min_complexity = min(self.item_complexity_map.values())

        if random_level < 0.0 or random_level > 1.0:
            raise ValueError(
                "Invalid random level {}, shoule be in the range [0.0, 1.0]".format(
                    random_level
                )
            )

        self.random_number = int((max_complexity - min_complexity) * random_level + 1)

    def shuffle_chunks(self):
        def chunks_wrap_padding(lst, n):
            """Yield successive n-sized chunks from lst."""
            num_chunks = max(1, self.num_samples)
            num_elements = num_chunks * n
            current_lst = []
            for i in range(num_elements):
                current_lst.append(lst[i % len(lst)])
                if len(current_lst) == n:
                    yield current_lst
                    current_lst = []

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            if self.random_number > 0:
                item_complexity_map = self.item_complexity_map.copy()
                complexity_random_ints = torch.randint(
                    self.random_number, (len(item_complexity_map),), generator=g
                ).tolist()

                for k, v in zip(item_complexity_map, complexity_random_ints):
                    item_complexity_map[k] += v

                ordered_item_complexity_map = OrderedDict(
                    sorted(item_complexity_map.items(), key=lambda t: t[1])
                )
            else:
                ordered_item_complexity_map = self.ordered_item_complexity_map

            index_chunks = list(
                chunks_wrap_padding(
                    list(ordered_item_complexity_map.keys()), self.num_replicas
                )
            )

            chunk_indices = torch.randperm(len(index_chunks), generator=g).tolist()  # type: ignore
        else:
            index_chunks = list(
                chunks_wrap_padding(
                    list(self.ordered_item_complexity_map.keys()), self.num_replicas
                )
            )
            chunk_indices = list(range(len(index_chunks)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.num_samples - len(chunk_indices)
            if padding_size <= len(chunk_indices):
                chunk_indices += chunk_indices[:padding_size]
            else:
                chunk_indices += (
                    chunk_indices * math.ceil(padding_size / len(chunk_indices))
                )[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            chunk_indices = chunk_indices[: self.num_samples]
        assert len(chunk_indices) == self.num_samples
        return index_chunks, chunk_indices

    def __iter__(self) -> Iterator:
        index_chunks, chunk_indices = self.shuffle_chunks()
        # subsample
        indices = [index_chunks[i][self.rank] for i in chunk_indices]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class LoadBalancingDistributedBatchSampler(Sampler):
    r"""Wraps another load balance sampler to yield variable sized mini-batches.

    Args:
        sampler (LoadBalancingDistributedSampler): Load balance sampler.
        batch_fn (Callable): Callable to yield mini-batch indices.
        drop_last (bool): If ``True``, the sampler will drop the last few batches exceeding
            the least number of batches among replicas, otherwise, the number of batches
            on each replica will be padded to the same.

    :attr:`batch_fn` will have the signature of::

        def batch_fn(indices: List[int]) -> List[List[int]]


    Example::
        >>> from bagua.torch_api.contrib import LoadBalancingDistributedSampler, \
        ...     LoadBalancingDistributedBatchSampler
        >>>
        >>> sampler = LoadBalancingDistributedSampler(dataset, complexity_fn=complexity_fn)
        >>> batch_sampler = LoadBalancingDistributedBatchSampler(sampler, batch_fn=batch_fn)
        >>> loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
        >>>
        >>> for epoch in range(start_epoch, n_epochs):
        ...     batch_sampler.set_epoch(epoch)
        ...     train(loader)

    """

    def __init__(
        self,
        sampler: LoadBalancingDistributedSampler,
        batch_fn,
        drop_last: bool = False,
    ) -> None:
        if not isinstance(sampler, LoadBalancingDistributedSampler):
            raise ValueError(
                "sampler should be of LoadBalancingDistributedSampler type."
            )

        if sampler.drop_last:
            raise ValueError("drop_last of sampler should be False")

        self.sampler = sampler
        self.batch_fn = batch_fn
        self.drop_last = drop_last

        self.num_replicas = self.sampler.num_replicas
        self.rank = self.sampler.rank

        self.generate_batches()

    def generate_batches(self):
        index_chunks, chunk_indices = self.sampler.shuffle_chunks()

        batches = []
        for rank in range(self.num_replicas):
            sub_indices = [index_chunks[i][rank] for i in chunk_indices]
            batches.append(self.batch_fn(sub_indices))

        self.total_batch = (
            max([len(b) for b in batches])
            if not self.drop_last
            else min([len(b) for b in batches])
        )

        # here {len(batches[self.rank]) - self.total_batch} batches dropped for
        # rank {self.rank}
        if self.total_batch < len(batches[self.rank]):
            pass

        self.padded_batches = [
            batch + batch[: self.total_batch - len(batch)] for batch in batches
        ]

    def __iter__(self):
        return iter(self.padded_batches[self.rank])

    def __len__(self):
        return self.total_batch

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.generate_batches()
