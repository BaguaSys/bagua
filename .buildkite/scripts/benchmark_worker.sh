#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

SYNTHETIC_SCRIPT="/bagua/examples/benchmark/synthetic_benchmark.py"

function check_benchmark_log {
    logfile=$1

    final_img_per_sec=$(cat ${logfile} | grep "Img/sec per " | tail -n 1 | awk '{print $4}')
    threshold="70.0"

    if [[ $final_img_per_sec -le $threshold ]]; then
        exit 1
    fi
}

export HOME=/workdir
cd /workdir && pip install . && git clone https://github.com/BaguaSys/examples.git
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
source $HOME/.cargo/env
pip install git+https://github.com/BaguaSys/bagua-core@master

logfile=$(mktemp /tmp/bagua_benchmark.XXXXXX.log)
python -m bagua.distributed.launch \
    --nnodes=2 \
    --nproc_per_node 4 \
    --node_rank=1 \
    --master_addr="10.158.66.134" \
    --master_port=1234 \
    ${SYNTHETIC_SCRIPT} \
    --num-iters 100 \
    --deterministic \
    2>&1 | tee ${logfile}
#check_benchmark_log ${logfile}
