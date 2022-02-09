import ctypes
import os
import pathlib


def _preload_libraries():
    libnccl_path = os.path.join(
        pathlib.Path.home(), ".local", "share", "bagua", "nccl", "lib", "libnccl.so"
    )
    if os.path.exists(libnccl_path):
        ctypes.CDLL(libnccl_path)
