#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail
hostname -i

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
MNIST_SCRIPT="/workdir/examples/mnist/main.py"

pip install git+https://github.com/BaguaSys/bagua-core@master

logfile=$(mktemp /tmp/bagua_mnist.XXXXXX.log)
python -m bagua.distributed.launch \
    --nnodes=2 \
    --nproc_per_node 2 \
    --node_rank=0 \
    --master_addr="23.236.107.69" \
    --master_port=1234 \
    ${MNIST_SCRIPT} \
    --set-deterministic \
    2>&1 | tee ${logfile}
#check_benchmark_log ${logfile}
