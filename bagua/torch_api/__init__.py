#!/usr/bin/env python3
"""
The Bagua communication library PyTorch interface.
"""
from . import distributed, tensor
from .communication import (  # noqa: F401
    init_process_group,
    allreduce,
    broadcast,
)
from .env import (
    get_rank,
    get_world_size,
    get_local_rank,
    get_local_size,
)
from . import contrib

import gorilla
patches = gorilla.find_patches([distributed, tensor])
for patch in patches:
    gorilla.apply(patch)

