#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

pip uninstall -y bagua bagua-core
export HOME=/workdir && cd $HOME
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
source $HOME/.cargo/env
# cd /workdir && python3 -m pip install --force-reinstall --no-cache-dir . || exit 1
cd /workdir && python3 setup.py install -f || exit 1
rm -rf bagua bagua_core
