#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

SYNTHETIC_SCRIPT="/workdir/examples/benchmark/synthetic_benchmark.py"

function check_benchmark_log {
    logfile=$1

    final_batch_loss=$(cat ${logfile} | grep "TrainLoss" | tail -n 1 | awk '{print $4}')
    final_img_per_sec=$(cat ${logfile} | grep "Img/sec per " | tail -n 1 | awk '{print $4}')

    python -c "import sys; sys.exit(1) if float($final_img_per_sec) != 0.001848"

    speed_threshold="1500.0"
    python -c "import sys; sys.exit(0 if float($final_img_per_sec) > float($threshold) else 1)"

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
    --node_rank=0 \
    --master_addr="10.158.66.134" \
    --master_port=1234 \
    ${SYNTHETIC_SCRIPT} \
    --num-iters 100 \
    --deterministic \
    2>&1 | tee ${logfile}
check_benchmark_log ${logfile}
