#!/usr/bin/env python3

from .base import Algorithm, AlgorithmImpl, GlobalAlgorithmRegistry  # noqa: F401
from . import bytegrad, decentralized, gradient_allreduce  # noqa: F401
from . import q_adam, async_model_average  # noqa: F401


GlobalAlgorithmRegistry.register(
    "gradient_allreduce",
    gradient_allreduce.GradientAllReduceAlgorithm,
    description="Gradient AllReduce Algorithm",
)
GlobalAlgorithmRegistry.register(
    "bytegrad", bytegrad.ByteGradAlgorithm, description="ByteGrad Algorithm"
)
GlobalAlgorithmRegistry.register(
    "decentralized",
    decentralized.DecentralizedAlgorithm,
    description="Decentralized SGD Algorithm",
)
GlobalAlgorithmRegistry.register(
    "low_precision_decentralized",
    decentralized.LowPrecisionDecentralizedAlgorithm,
    description="Low Precision Decentralized SGD Algorithm",
)
GlobalAlgorithmRegistry.register(
    "qadam", q_adam.QAdamAlgorithm, description="QAdam Algorithm"
)
GlobalAlgorithmRegistry.register(
    "async",
    async_model_average.AsyncModelAverageAlgorithm,
    description="Asynchronous Model Average Algorithm",
)
