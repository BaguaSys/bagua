#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.distributed import BaguaModule
from typing import List
import torch
from bagua.torch_api.algorithms_implementation.decentralized_implementation import (
    DecentralizedAlgorithm_Implementation,
    LowPrecisionDecentralizedAlgorithm_Implementation,
)


class DecentralizedAlgorithm:
    def __init__(
        self,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
        communication_interval: int = 1,
    ):
        """
        Create an instance of the
        `Decentralized SGD <https://bagua-tutorials.kwai-seattle.com/algorithms/decentralized>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            peer_selection_mode (str): Can be ``"all"`` or ``"shift_one"``. ``"all"`` means all workers'
                weights are averaged in each communication step. ``"shift_one"`` means each worker
                selects a different peer to do weights average in each communication step.
            communication_interval (int): Number of iterations between two communication steps.

        """
        self.hierarchical = hierarchical
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval

    def reify(self) -> DecentralizedAlgorithm_Implementation:
        return DecentralizedAlgorithm_Implementation(
            hierarchical=self.hierarchical,
            peer_selection_mode=self.peer_selection_mode,
            communication_interval=self.communication_interval,
        )


class LowPrecisionDecentralizedAlgorithm:
    def __init__(self, hierarchical: bool = True, communication_interval: int = 1):
        """
        Create an instance of the
        `Low Precision Decentralized SGD <https://bagua-tutorials.kwai-seattle.com/algorithms/low-precision-decentralized>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
        """
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
    
    def reify(self) -> LowPrecisionDecentralizedAlgorithm_Implementation:
        return LowPrecisionDecentralizedAlgorithm_Implementation(
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
        )
