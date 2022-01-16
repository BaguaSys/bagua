#!/bin/sh
set -eux

exit_and_error() {
    echo "Auto installation is supported only on Ubuntu(18.04) or CentOs(7,8), abort."
    exit
}

check_os_version() {
    OS_NAME=$(grep ^NAME /etc/os-release | awk -F'"' '{print $2}')
    VERSION_ID=$(grep ^VERSION_ID /etc/os-release | awk -F'"' '{print $2}')
    echo "Current OS is "${OS_NAME}", Version is "${VERSION_ID}
    if [ "$OS_NAME" == "Ubuntu" ]; then
        if [[ $VERSION_ID != @("18.04") ]]; then
            exit_and_error
        fi
    elif [ "$OS_NAME" == "CentOS Linux" ]; then
        if [[ $VERSION_ID != @("7"|"8") ]]; then
            exit_and_error
        fi
    else
        exit_and_error
    fi
}

# upgrade to python3.8
confirm() {
    # call with a prompt string or use a default
    echo "Your Python version is $(python3 -V), but Bagua requires Python version >= 3.7."
    read -r -p "${1:-Do you want to upgrade Python? [Y/n]} " response
    case "$response" in
    [yY][eE][sS] | [yY])
        echo "True"
        ;;
    *)
        echo "False"
        ;;
    esac

}

upgrade_python() {
    if [ "$OS_NAME" == "Ubuntu" ]; then
        apt-get install -y python3.8 python3.8-distutils python3.8-dev
    elif [ "$OS_NAME" == "CentOS Linux" ]; then
        mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz &&
            tar xvf /var/tmp/Python-3.8.12.tgz &&
            cd /var/tmp/Python-3.8.12 && ./configure --enable-optimizations --prefix=/usr && make altinstall &&
            rm -rf /var/tmp/Python-3.8.12.tgz /var/tmp/Python-3.8.12 && cd -
    fi
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
}

check_python_version() {
    PYTHON_VERSION_OK=$(python3 -c 'import sys; print(int(sys.version_info > (3, 7)))')
    if [[ $PYTHON_VERSION_OK -eq 0 ]]; then
        confirm && upgrade_python
    fi

}

check_os_version

# install necessary packages
if [ "$OS_NAME" == "Ubuntu" ]; then
    # remove cmake
    apt remove --purge --auto-remove -y cmake

    # install python3-pip
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl software-properties-common wget
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip zlib1g-dev libssl-dev

elif [ "$OS_NAME" == "CentOS Linux" ]; then
    if [ $VERSION_ID == "7" ]; then
        yum remove cmake3 -y
    elif [ $VERSION_ID == "8" ]; then
        yum remove cmake -y
    fi

    yum install -y wget curl bzip2 perl zlib-devel openssl-devel
fi

check_python_version

# install some utils
python3 -m pip install --upgrade pip -i https://pypi.org/simple
python3 -m pip install setuptools-rust colorama tqdm wheel -i https://pypi.org/simple

# install cmake 3.22.1
mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh &&
    cd /var/tmp && chmod +x cmake-3.22.1-linux-x86_64.sh &&
    sh cmake-3.22.1-linux-x86_64.sh --prefix=/usr --skip-license &&
    rm -rf /var/tmp/cmake-3.22.1-linux-x86_64.sh && cd -

# install hwloc 2.7.0
mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://download.open-mpi.org/release/hwloc/v2.7/hwloc-2.7.0.tar.bz2 &&
    tar -x -f /var/tmp/hwloc-2.7.0.tar.bz2 -C /var/tmp -j &&
    cd /var/tmp/hwloc-2.7.0 && ./configure &&
    make -j$(nproc) &&
    make -j$(nproc) install &&
    rm -rf /var/tmp/hwloc* && cd -

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH

# install openmpi 4.1.2
mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.2.tar.bz2 &&
    tar -x -f /var/tmp/openmpi-4.1.2.tar.bz2 -C /var/tmp -j &&
    cd /var/tmp/openmpi-4.1.2 && ./configure --disable-getpwuid --disable-oshmem --enable-fortran --enable-mca-no-build=btl-uct --enable-orterun-prefix-by-default --with-cuda --without-verbs &&
    make -j$(nproc) &&
    make -j$(nproc) install &&
    rm -rf /var/tmp/openmpi-4.1.2 /var/tmp/openmpi-4.1.2.tar.bz2 && cd -

# install rust
if ! command -v cargo &>/dev/null; then
    curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# develop version (git master)
python3 -m pip install --pre bagua --upgrade
