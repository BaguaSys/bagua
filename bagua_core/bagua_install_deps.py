import os
import platform
import shutil
import tempfile
import urllib.request
import pathlib
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
    _make_nccl_record("11.5", "2.11.4", "2.11", "nccl_2.11.4-1+cuda11.4_x86_64.txz")
)
_nccl_records.append(
    _make_nccl_record("11.4", "2.11.4", "2.11", "nccl_2.11.4-1+cuda11.4_x86_64.txz")
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
    _make_nccl_record("11.0", "2.11.4", "2.11", "nccl_2.11.4-1+cuda11.0_x86_64.txz")
)
_nccl_records.append(
    _make_nccl_record("10.2", "2.11.4", "2.11", "nccl_2.11.4-1+cuda10.2_x86_64.txz")
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

    destination = os.path.join(prefix, "nccl")

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


def install_deps():
    nvcc_version = (
        os.popen(
            "nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'"
        )
        .read()
        .strip()
    )
    print("nvcc_version: ", nvcc_version)
    install_lib(
        nvcc_version,
        os.path.join(pathlib.Path.home(), ".local", "share", "bagua"),
        "nccl",
    )


if __name__ == "__main__":
    install_deps()
