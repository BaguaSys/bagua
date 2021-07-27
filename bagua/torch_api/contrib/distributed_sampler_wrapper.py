# This is based on https://github.com/catalyst-team/catalyst/blob/ce79bbecf0ceb09f972c7cc7ebb9974d1011f17b/catalyst/data/sampler.py.

from torch.utils.data import Dataset, Sampler, DistributedSampler
from typing import Iterator, Optional
from operator import itemgetter


class SamplerDataset(Dataset):
    r"""See the indexes from `Sampler` as an individual dataset.

    Arguments:
        sampler: Pytorch sampler.
    """

    def __init__(self, sampler: Sampler):
        self.sampler_dataset = sampler
        self.sampler_dataset_list = None

    def __getitem__(self, index: int):
        r"""Fetch elements from dataset.
        Arguments:
            index: index of the element in the dataset.
        Returns:
            Single element by index.
        """
        if self.sampler_dataset_list is None:
            self.sampler_dataset_list = list(self.sampler_dataset)
        return self.sampler_dataset_list[index]

    def __len__(self) -> int:
        r"""
        Returns:
            Length of the dataset.
        """
        return len(self.sampler_dataset)


class DistributedSamplerWrapper(DistributedSampler):
    r"""
    A sampler wrapper attaches the original sampler with distributed feature.
    With this feature you can use any sampler in distributed mode.
    This is intended for the scenario where a dataset sampler need to be used in
    distributed algorithm and usually appeared before `model.with_bagua()`.

    Arguments:
        sampler: Sampler used for subsampling. It can be any other pytorch
            sampler except for `DistributedSampler`.
        num_replicas (int, optional): Total number of processes in
            distributed training.
        rank (int, optional): Global rank of the current process
            within ``num_replicas``.
        shuffle (bool, optional): If true, sampler will shuffle the indices.
            default: `True`.

    .. note::
        Sampler is assumed to be of constant size.

    Examples::
        >>> train_dataset = MyDataset(...)
        >>> weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(...)
        >>> from bagua.torch_api.contrib.distributed_sampler_wrapper import DistributedSamplerWrapper
        >>> train_sampler = DistributedSamplerWrapper(weighted_sampler,
                                            num_replicas = bagua.get_world_size(),
                                            rank = bagua.get_rank()
                                            )
        >>> train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = args.batch_size,
                                                    shuffle = False,
                                                    num_workers = args.num_workers,
                                                    pin_memory = False,
                                                    sampler = train_sampler)
        ...
        ...
        ...
        >>> model = model.with_bagua(
            ...      [optimizer],
            ...      GradientAllReduce()
            ...    )
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        super(DistributedSamplerWrapper, self).__init__(
            SamplerDataset(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        r"""Iterator over sampler.
        Returns:
            Python iterator generating batch samples indexes.
        """
        # index for distributed mode sampled from original index set.
        indexes_of_indexes = super().__iter__()
        # see the data in sampler as a subdataset.
        sampler_subdataset_indexes = SamplerDataset(self.sampler)
        return iter(itemgetter(*indexes_of_indexes)(sampler_subdataset_indexes))
