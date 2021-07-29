#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

nvidia-smi

SYNTHETIC_SCRIPT="/bagua/examples/benchmark/synthetic_benchmark.py"

function parse_benchmark_log {
    logfile=$1

    img_per_sec=$(cat ${logfile} | grep "Img/sec per " | tail -n 1 | awk '{print $5}')

    echo $img_per_sec
}

logfile=$(mktemp /tmp/bagua_benchmark.XXXXXX.log)
python -m bagua.distributed.run \
    --autotune_level 1 \
    --nproc_per_node 8 \
    --no_python \
    python ${SYNTHETIC_SCRIPT} \
    2>&1 | tee ${logfile}
parse_benchmark_log ${logfile}
