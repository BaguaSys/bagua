#!/bin/bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euo pipefail
cp -a /upstream /workdir
# setuptools will reset "$HOME" to the setup.py execution directory, set dependency of rust at new "$HOME" to ensure that rustc can run normally.
ln -s $HOME/.rustup /workdir
ln -s $HOME/.cargo /workdir
export HOME=/workdir && cd $HOME && bash .buildkite/scripts/install_bagua.sh || exit 1
pip install pytest-timeout
pip install git+https://github.com/PyTorchLightning/pytorch-lightning.git
pytest --timeout=300 -s -o "testpaths=tests"
