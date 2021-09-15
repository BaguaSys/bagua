import ctypes
import os


def _preload_libraries():
    cwd = os.path.dirname(os.path.abspath(__file__))
    libnccl_path = os.path.join(cwd, ".data", "lib", "libnccl.so")
    ctypes.CDLL(libnccl_path)
