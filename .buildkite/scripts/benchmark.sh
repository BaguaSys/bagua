#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

function finish {
    rm -rf $(find /workdir -group root)
}
trap finish EXIT

SYNTHETIC_SCRIPT="/bagua/examples/benchmark/synthetic_benchmark.py"

function parse_benchmark_log {
    logfile=$1

    img_per_sec=$(cat ${logfile} | grep "Img/sec per " | tail -n 1 | awk '{print $4}')

    echo $img_per_sec
}

pip install --upgrade --force-reinstall git https://github.com/shjwudp/bagua-core.git@telemetry

logfile=$(mktemp /tmp/bagua_benchmark.XXXXXX.log)
python -m bagua.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node 4 \
    --no_python \
    --autotune_level 1 \
    python ${SYNTHETIC_SCRIPT} \
        --num-batches-per-iter 100 \
        2>&1 | tee ${logfile}
parse_benchmark_log ${logfile}
