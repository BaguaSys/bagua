#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

pip uninstall -y bagua bagua-core
git config --global --add safe.directory /workdir/rust/bagua-core/bagua-core-internal/third_party/Aluminum
cd /workdir && python3 setup.py install -f || exit 1
rm -rf bagua bagua_core
