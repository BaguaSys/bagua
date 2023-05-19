#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

SOURCE_DIR=$1
WORK_DIR=$2

cp -a $SOURCE_DIR $WORK_DIR
# setuptools will reset "$HOME" to the setup.py execution directory, set dependency of rust at new "$HOME" to ensure that rustc can run normally.
ln -s $HOME/.rustup $WORK_DIR
ln -s $HOME/.cargo $WORK_DIR

pip uninstall -y bagua bagua-core
git config --global --add safe.directory $WORK_DIR/rust/bagua-core/bagua-core-internal/third_party/Aluminum
cd $WORK_DIR && python3 setup.py install -f || exit 1
rm -rf bagua bagua_core
