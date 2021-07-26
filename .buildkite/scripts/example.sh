#!/bin/bash

set -euo pipefail

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"
nvidia-smi
