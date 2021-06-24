#!/bin/sh
set -eu

exit_and_error() {
  echo "Auto installation is supported only on Ubuntu(16.04,18.04,20.04) or CentOs(7,8), abort."
  exit
}

check_os_version() {
  OS_NAME=`grep ^NAME /etc/os-release | awk -F'"' '{print $2}'`
  VERSION_ID=`grep ^VERSION_ID /etc/os-release | awk -F'"' '{print $2}'`
  echo "Current OS is "${OS_NAME}", Version is "${VERSION_ID}
  if [ $OS_NAME == "Ubuntu" ]; then
    if [[ $VERSION_ID != @("16.04"|"18.04"|"20.04") ]]; then
      exit_and_error
    fi
  elif [ $OS_NAME == "CentOS Linux" ]; then
    if [[ $VERSION_ID != @("7"|"8") ]]; then
      exit_and_error
    fi
  else
    exit_and_error
  fi
}


check_os_version

if [ $OS_NAME == "Ubuntu" ]; then
  # install cmake & python3-pip
  apt-get update && apt-get install -y curl software-properties-common wget
  wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
  if [ $VERSION_ID == "16.04" ]; then
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'
  elif [ $VERSION_ID == "18.04" ]; then
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
  elif [ $VERSION_ID == "20.04" ]; then
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
  fi
  apt-get update && apt-get install -y cmake python3-pip

  # install some utils
  python3 -m pip install setuptools-rust colorama tqdm

  # install rust
  curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y

  # install zlib, ssl, openmpi
  apt-get install -y zlib1g-dev libssl-dev openmpi-bin openmpi-doc libopenmpi-dev

  # install bagua
  python3 -m pip install bagua -f https://repo.arrayfire.com/python/wheels/3.8.0/
elif [ $OS_NAME == "CentOS Linux" ]; then
  yum install -y cmake3
fi
