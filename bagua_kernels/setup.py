import os
import torch
import subprocess
import pathlib
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils import cpp_extension

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                        universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")

def load_fused_add_dropout_residual_kernel():

    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')

    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / 'build'

    create_build_dir(buildpath)

    fused_dropout_add_cuda = cpp_extension.load(
        name='add_dropout_residual_impl',
        sources=[srcpath / 'add_dropout_residual.cpp',
                 srcpath / 'add_dropout_residual.cu'],
        build_directory=buildpath,
        extra_cflags=['-O3',],
        extra_cuda_cflags=['-O3',
                           '-gencode', 'arch=compute_70,code=sm_70',
                           '-U__CUDA_NO_HALF_OPERATORS__',
                           '-U__CUDA_NO_HALF_CONVERSIONS__',
                           '--expt-relaxed-constexpr',
                           '--expt-extended-lambda',
                           '--use_fast_math'] + cc_flag)
