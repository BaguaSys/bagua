import os
from setuptools import setup, find_packages
from distutils.errors import (
    DistutilsPlatformError,
)
from setuptools_rust import Binding, RustExtension
import platform
import shutil
import sys
import tempfile
import urllib.request
from tqdm import tqdm


_nccl_records = []
library_records = {}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def _make_nccl_url(public_version, filename):
    # https://developer.download.nvidia.com/compute/redist/nccl/v2.8/nccl_2.8.4-1+cuda11.2_x86_64.txz
    return (
        "https://developer.download.nvidia.com/compute/redist/nccl/"
        + "v{}/{}".format(public_version, filename)
    )


def _make_nccl_record(cuda_version, full_version, public_version, filename_linux):
    return {
        "cuda": cuda_version,
        "nccl": full_version,
        "assets": {
            "Linux": {
                "url": _make_nccl_url(public_version, filename_linux),
                "filename": "libnccl.so.{}".format(full_version),
            },
        },
    }


_nccl_records.append(
    _make_nccl_record("11.4", "2.10.3", "2.10", "nccl_2.10.3-1+cuda11.4_x86_64.txz")
)
_nccl_records.append(
    _make_nccl_record("11.3", "2.10.3", "2.10", "nccl_2.10.3-1+cuda11.0_x86_64.txz")
)
_nccl_records.append(
    _make_nccl_record("11.2", "2.10.3", "2.10", "nccl_2.10.3-1+cuda11.0_x86_64.txz")
)
_nccl_records.append(
    _make_nccl_record("11.1", "2.10.3", "2.10", "nccl_2.10.3-1+cuda11.0_x86_64.txz")
)
_nccl_records.append(
    _make_nccl_record("11.0", "2.10.3", "2.10", "nccl_2.10.3-1+cuda11.0_x86_64.txz")
)
_nccl_records.append(
    _make_nccl_record("10.2", "2.10.3", "2.10", "nccl_2.10.3-1+cuda10.2_x86_64.txz")
)
_nccl_records.append(
    _make_nccl_record("10.1", "2.10.3", "2.10", "nccl_2.10.3-1+cuda10.2_x86_64.txz")
)
library_records["nccl"] = _nccl_records


def install_baguanet(url, destination):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, os.path.basename(url))
        print("Downloading {}...".format(url))
        download_url(url, filename)
        outdir = os.path.join(tmpdir, "extract")
        shutil.unpack_archive(filename, outdir)
        lib_dir = os.path.join(outdir, "build")
        for filename in os.listdir(lib_dir):
            shutil.move(os.path.join(lib_dir, filename), destination)


def install_lib(cuda, prefix, library):
    record = None
    lib_records = library_records
    for record in lib_records[library]:
        if record["cuda"] == cuda:
            break
    else:
        raise RuntimeError(
            """
The CUDA version({}) specified is not supported.
Should be one of {}.""".format(
                cuda, str([x["cuda"] for x in lib_records[library]])
            )
        )
    if prefix is None:
        prefix = os.path.expanduser("~/.bagua_core/cuda_lib")
    destination = calculate_destination(prefix, cuda, library, record[library])

    if os.path.exists(destination):
        print("The destination directory {} already exists.".format(destination))
        shutil.rmtree(destination)

    target_platform = platform.system()
    asset = record["assets"].get(target_platform, None)
    if asset is None:
        raise RuntimeError(
            """
The current platform ({}) is not supported.""".format(
                target_platform
            )
        )

    print(
        "Installing {} {} for CUDA {} to: {}".format(
            library, record[library], record["cuda"], destination
        )
    )

    url = asset["url"]
    print("Downloading {}...".format(url))
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, os.path.basename(url))
        download_url(url, filename)
        print("Extracting...")
        outdir = os.path.join(tmpdir, "extract")
        shutil.unpack_archive(filename, outdir)
        print("Installing...")
        if library == "nccl":
            subdir = os.listdir(outdir)
            assert len(subdir) == 1
            shutil.move(os.path.join(outdir, subdir[0]), destination)

            # Install bagua-net
            dst_dir = os.path.join(destination, "bagua-net")
            os.mkdir(dst_dir)
            install_baguanet(
                "https://github.com/BaguaSys/bagua-net/releases/download/v0.1.1/bagua-net_refs.tags.v0.1.1_x86_64.tar.gz",
                dst_dir,
            )
        else:
            assert False
        print("Cleaning up...")
    print("Done!")


def calculate_destination(prefix, cuda, lib, lib_ver):
    """Calculates the installation directory."""
    return os.path.join(prefix, ".data")


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
    install_lib(nvcc_version, os.path.join(cwd, "bagua_core"), "nccl")


if __name__ == "__main__":
    import colorama

    colorama.init(autoreset=True)
    cwd = os.path.dirname(os.path.abspath(__file__))

    if (
        int(os.getenv("BAGUA_NO_INSTALL_DEPS", 0)) == 0
        and len(sys.argv) > 1
        and sys.argv[1] in ["install", "bdist_wheel"]
    ):
        print(
            colorama.Fore.BLACK
            + colorama.Back.CYAN
            + "Bagua is automatically installing some system dependencies like NCCL, to disable set env variable BAGUA_NO_INSTALL_DEPS=1",
        )
        install_dependency_library()

    setup(
        name="bagua",
        use_scm_version={"local_scheme": "no-local-version"},
        setup_requires=["setuptools_scm"],
        url="https://github.com/BaguaSys/bagua",
        python_requires=">=3.7",
        description="Bagua is a flexible and performant distributed training algorithm development framework.",
        packages=find_packages(exclude=("tests")),
        package_data={
            "": [
                ".data/lib/libnccl.so",
                ".data/bagua-net/libbagua_net.so",
                ".data/bagua-net/libnccl-net.so",
            ]
        },
        rust_extensions=[
            RustExtension(
                "bagua_core.bagua_core",
                path="rust/bagua-core/bagua-core-py/Cargo.toml",
                binding=Binding.PyO3,
                native=False,
            )
        ],
        author="Kuaishou AI Platform & DS3 Lab",
        author_email="admin@mail.xrlian.com",
        install_requires=[
            "setuptools_rust",
            "colorama",
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
            "xxhash==v2.0.2",
        ],
        entry_points={
            "console_scripts": [
                "baguarun = bagua.script.baguarun:main",
            ],
        },
        scripts=[
            "bagua/script/bagua_sys_perf",
        ],
        zip_safe=False,
    )
