#!/bin/bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euo pipefail

export HOME=/workdir
cd /workdir
pytest
