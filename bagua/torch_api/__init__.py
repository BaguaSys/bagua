#!/usr/bin/env python3
"""
The Bagua communication library PyTorch interface.
"""
from distutils.errors import (
    DistutilsPlatformError,
)

try:
    import torch
except ImportError:
    print("import torch failed, is it installed?")

version = torch.__version__
if version is None:
    raise DistutilsPlatformError(
        "Unable to determine PyTorch version from the version string '%s'"
        % torch.__version__
    )
elif version < "1.6.0":
    raise Exception(
        "Bagua need pytorch version >= 1.6.0, while current version is {}.".format(
            version
        )
    )

from .communication import (  # noqa: F401
    get_backend,
    init_process_group,
    send,
    recv,
    broadcast,
    reduce,
    reduce_inplace,
    gather,
    gather_inplace,
    scatter,
    scatter_inplace,
    allreduce,
    allreduce_inplace,
    allgather,
    allgather_inplace,
    alltoall,
    alltoall_inplace,
    alltoall_v,
    alltoall_v_inplace,
    reduce_scatter,
    reduce_scatter_inplace,
    ReduceOp,
)
from .distributed import BaguaModule  # noqa: F401
from .tensor import BaguaTensor  # noqa: F401
from .env import (  # noqa: F401
    get_rank,
    get_world_size,
    get_local_rank,
    get_local_size,
)
from . import contrib  # noqa: F401
from . import communication  # noqa: F401
from . import algorithms  # noqa: F401
from . import checkpoint  # noqa: E402,F401
from . import data_parallel  # noqa: E402,F401
from .model_parallel import moe  # noqa: E402,F401
