#!/usr/bin/env python3

from typing import List


class Algorithm:
    def __init__(
        self,
    ):
        pass

    def need_reset(self) -> bool:
        "return True when we need to call init_buckets, init_hooks again. for example when we collect more info and want to rearrange the buckets"
        # TODO: previous buckets and hooks need to be cleared before reinit
        pass

    def init_buckets(self, module, optimizer) -> List:
        pass

    def init_hooks(self, module, optimizer) -> List:
        pass

    def init_operations(
        self,
        bucket,
        inter_node_communicator,
        intra_node_communicator,
        global_communicator,
    ):
        pass


class DevelopAlgoritm(Algorithm):
    def init_buckets(self, module, optimizer) -> List:
        # TODO: finish me @shaoduo
        ...
