#!/bin/bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euo pipefail

bash install_bagua.sh /upstream /workdir || exit 1
pip install pytest-timeout
pip install git+https://github.com/PyTorchLightning/pytorch-lightning.git
pytest --timeout=300 -s -o "testpaths=tests"
