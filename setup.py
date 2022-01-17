import os
from setuptools import setup, find_packages
from distutils.errors import (
    DistutilsPlatformError,
)
from setuptools_rust import Binding, RustExtension
import shutil
import sys
from typing import List
from pathlib import Path

library_records = {}


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


def install_baguanet(destination):
    os.makedirs(destination, exist_ok=True)
    os.system("cd rust/bagua-net/cc && make")
    shutil.move(
        "rust/bagua-net/cc/libnccl-net.so", os.path.join(destination, "libnccl-net.so")
    )


def install_dependency_library():
    nvcc_version = (
        os.popen(
            "nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'"
        )
        .read()
        .strip()
    )
    print("nvcc_version: ", nvcc_version)
    install_baguanet(os.path.join(cwd, "bagua_core", ".data", "bagua-net"))


if __name__ == "__main__":
    import colorama

    colorama.init(autoreset=True)
    cwd = os.path.dirname(os.path.abspath(__file__))

    def check_args(args: List[str]) -> bool:
        for arg in ["build", "install", "develop", "bdist_wheel", "wheel"]:
            if arg in args:
                return True
        return False

    if (
        int(os.getenv("BAGUA_NO_INSTALL_DEPS", 0)) == 0
        and len(sys.argv) > 1
        and check_args(sys.argv)  # noqa: W503
    ):
        print(
            colorama.Fore.BLACK
            + colorama.Back.CYAN
            + "Bagua is automatically installing some system dependencies like bagua-net, to disable set env variable BAGUA_NO_INSTALL_DEPS=1",
        )
        os.system("python3 bagua_core/bagua_install_deps.py")
        install_dependency_library()

    name_suffix = os.getenv("BAGUA_CUDA_VERSION", "")
    if name_suffix != "":
        name_suffix = "-cuda" + name_suffix

    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text()

    setup(
        name="bagua" + name_suffix,
        version="0.9.0",
        url="https://github.com/BaguaSys/bagua",
        python_requires=">=3.7",
        description="Bagua is a deep learning training acceleration framework for PyTorch. It provides a one-stop training acceleration solution, including faster distributed training compared to PyTorch DDP, faster dataloader, kernel fusion, and more.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=("tests")),
        package_data={
            "": [
                ".data/bagua-net/libnccl-net.so",
            ]
        },
        rust_extensions=[
            RustExtension(
                "bagua_core.bagua_core",
                path="rust/bagua-core/bagua-core-py/Cargo.toml",
                binding=Binding.PyO3,
                native=False,
            ),
        ],
        author="Kuaishou AI Platform & DS3 Lab",
        author_email="admin@mail.xrlian.com",
        install_requires=[
            "setuptools_rust",
            "colorama",
            "tqdm",
            "deprecation>=2.1",
            "pytest-benchmark>=3.4",
            "scikit-optimize>=0.8.1",
            "scikit-learn>=0.24,<=1.0.1,!=1.0",
            "numpy",
            "flask>=2.0",
            "prometheus_client>=0.11",
            "parallel-ssh==2.8.0",
            "pydantic>=1.8",
            "requests>=2.25",
            "gorilla==0.4.0",
            "gevent>=21.8",
            "xxhash>=2.0",
        ],
        entry_points={
            "console_scripts": [
                "baguarun = bagua.script.baguarun:main",
            ],
        },
        scripts=["bagua/script/bagua_sys_perf", "bagua_core/bagua_install_deps.py"],
        zip_safe=False,
    )
