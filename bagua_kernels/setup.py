from setuptools import setup
import os
import torch
import subprocess
import pathlib
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils import cpp_extension

ext_modules = []

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                        universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(
        cuda_dir
    )
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (
        bare_metal_minor != torch_binary_minor
    ):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            + "not match the version used to compile Pytorch binaries.  "
            + "Pytorch binaries were compiled with Cuda {}.\n".format(
                torch.version.cuda
            )
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            + "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )

check_cuda_torch_binary_vs_bare_metal(cpp_extension.CUDA_HOME)

def build_ext_modules():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cc_flag = []
    _, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")

    ext_modules.append(
        cpp_extension.CUDAExtension(
            name='add_dropout_residual_impl',
            sources=['add_dropout_residual.cpp',
                     'add_dropout_residual_kernel.cu'],
            extra_compile_args={
                "cxx": [
                    "-O3",
                ],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_75,code=compute_75",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ]
                + cc_flag},))


if __name__ == "__main__":
    build_ext_modules()
    setup(
        name="bagua_kernels",
        ext_modules=ext_modules,
        cmdclass={"build_ext": cpp_extension.BuildExtension},
    )
