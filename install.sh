#!/bin/sh
set -eux

# install cmake & python3-pip
apt-get update && apt-get install -y curl software-properties-common wget
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
apt-get update && apt-get install -y cmake python3-pip

# install some utils
python3 -m pip install setuptools-rust colorama tqdm

# install rust
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y

# install zlib, ssl, openmpi
apt-get install -y zlib1g-dev libssl-dev openmpi-bin openmpi-doc libopenmpi-dev

# install bagua
python3 -m pip install bagua -f https://repo.arrayfire.com/python/wheels/3.8.0/
