import os
from distutils.errors import (
    DistutilsPlatformError,
)
from bagua_install_library import install_library
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension
import colorama
import sys


def check_torch_version():
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
    return version


def install_dependency_library():
    nvcc_version = (
        os.popen(
            "nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'"
        )
        .read()
        .strip()
    )
    print("nvcc_version: ", nvcc_version)
    args = [
        "--library",
        "nccl",
        "--cuda",
        nvcc_version,
        "--prefix",
        os.path.join(cwd, "bagua"),
    ]
    install_library.main(args)


if __name__ == "__main__":
    colorama.init(autoreset=True)
    cwd = os.path.dirname(os.path.abspath(__file__))

    if int(os.getenv("BAGUA_NO_INSTALL_DEPS", 0)) == 0:
        print(
            colorama.Fore.BLACK
            + colorama.Back.CYAN
            + "Bagua is automatically installing some system dependencies like NCCL, to disable set env variable BAGUA_NO_INSTALL_DEPS=1",
        )
        install_dependency_library()

    setup(
        name="bagua-core",
        version="0.1.0",
        url="https://github.com/BaguaSys/bagua-core",
        python_requires=">=3.6",
        description="Core communication lib for Bagua.",
        rust_extensions=[
            RustExtension(
                "bagua_core_py",
                path="bagua-core-py/Cargo.toml",
                binding=Binding.PyO3,
                native=True,
            )
        ],
        author="Kuaishou AI Platform & DS3 Lab",
        author_email="admin@mail.xrlian.com",
        package_data={"": [".data/lib/libnccl.so"]},
        install_requires=[],
    )
