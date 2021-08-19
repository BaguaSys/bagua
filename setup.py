import os
from setuptools import setup, find_packages
import sys


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))

    setup(
        name="bagua",
        use_scm_version={"local_scheme": "no-local-version"},
        setup_requires=["setuptools_scm"],
        url="https://github.com/BaguaSys/bagua",
        python_requires=">=3.7",
        description="Bagua is a flexible and performant distributed training algorithm development framework.",
        packages=find_packages(exclude=("tests")),
        author="Kuaishou AI Platform & DS3 Lab",
        author_email="admin@mail.xrlian.com",
        install_requires=[
            "bagua-core>=0.4.1,<0.5",
            "deprecation",
            "pytest-benchmark",
            "scikit-optimize",
            "numpy",
            "flask",
            "prometheus_client",
            "parallel-ssh",
            "pydantic",
            "requests",
            "gorilla",
            "gevent",
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
