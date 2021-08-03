import os
from distutils.errors import (
    DistutilsPlatformError,
)
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension
import sys
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
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
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
    install_lib(nvcc_version, os.path.join(cwd, "python/bagua_core"), "nccl")


if __name__ == "__main__":
    import colorama
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
        use_scm_version={"local_scheme": "no-local-version"},
        setup_requires=["setuptools_scm"],
        url="https://github.com/BaguaSys/bagua-core",
        python_requires=">=3.6",
        description="Core communication lib for Bagua.",
        package_dir={"": "python/"},
        packages=find_packages("python/"),
        package_data={"": [".data/lib/libnccl.so"]},
        rust_extensions=[
            RustExtension(
                "bagua_core.bagua_core",
                path="bagua-core-py/Cargo.toml",
                binding=Binding.PyO3,
                native=True,
            )
        ],
        author="Kuaishou AI Platform & DS3 Lab",
        author_email="admin@mail.xrlian.com",
        install_requires=[
            "setuptools_rust",
            "colorama",
        ],
        zip_safe=False,
    )
