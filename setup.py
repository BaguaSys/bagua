import os
from distutils.errors import (
    DistutilsPlatformError,
)
from setuptools import setup, find_packages
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


if __name__ == "__main__":
    colorama.init(autoreset=True)
    cwd = os.path.dirname(os.path.abspath(__file__))

    setup(
        name="bagua",
        version="0.1.0",
        url="https://github.com/BaguaSys/bagua",
        python_requires=">=3.6",
        description="Bagua is a flexible and performant distributed training algorithm development framework.",
        packages=find_packages(exclude=("tests")),
        author="Kuaishou AI Platform & DS3 Lab",
        author_email="admin@mail.xrlian.com",
        install_requires=[
            "bagua-core",
            "deprecation",
            "pytest-benchmark",
            "scikit-optimize",
            "numpy",
            "flask",
            "prometheus_client",
            "parallel-ssh",
            "pydantic",
        ],
        entry_points={
            "console_scripts": [
                "baguarun = bagua.script.baguarun:main",
            ],
        },
        scripts=[
            "bagua/script/bagua_sys_perf",
        ],
    )
