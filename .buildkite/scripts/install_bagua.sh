#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

pip uninstall -y bagua bagua-core
export HOME=/workdir && cd $HOME
cd /workdir && python3 setup.py install -f || exit 1
rm -rf bagua bagua_core
