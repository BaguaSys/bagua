import ctypes
import os
import logging
import pathlib


def _preload_libraries():
    libnccl_path = os.path.join(
        pathlib.Path.home(), ".local", "share", "bagua", "nccl", "lib", "libnccl.so"
    )
    if os.path.exists(libnccl_path):
        ctypes.CDLL(libnccl_path)
    else:
        logging.warning(
            "Bagua cannot detect bundled NCCL library, Bagua will try to use system NCCL instead. If you encounter any error, please run `import bagua_core; bagua_core.install_deps()` or the `bagua_install_deps.py` script to install bundled libraries."
        )
