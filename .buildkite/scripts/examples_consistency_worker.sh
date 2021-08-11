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
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
source $HOME/.cargo/env
pip install git+https://github.com/BaguaSys/bagua-core@master

#MNIST_SCRIPT="/workdir/examples/mnist/main.py"
logfile=$(mktemp /tmp/bagua_mnist.XXXXXX.log)
cd /workdir/examples/mnist
sleep 5d
python -m bagua.distributed.launch \
    --nnodes=2 \
    --nproc_per_node 2 \
    --node_rank=0 \
    --master_addr="10.158.66.134" \
    --master_port=1234 \
    main.py \
    --set-deterministic \
    2>&1 | tee ${logfile}
#check_benchmark_log ${logfile}
