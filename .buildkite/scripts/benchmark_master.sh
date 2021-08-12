#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

SYNTHETIC_SCRIPT="/workdir/examples/benchmark/synthetic_benchmark.py"

function check_benchmark_log {
    logfile=$1

    final_batch_loss=$(cat ${logfile} | grep "TrainLoss" | tail -n 1 | awk '{print $4}')
    final_img_per_sec=$(cat ${logfile} | grep "Img/sec per " | tail -n 1 | awk '{print $4}')

    python -c "import sys; sys.exit(1) if float($final_batch_loss) != 0.001848 else print('final_batch_loss is euqal.')"

    speed_threshold="200.0"
    python -c "import sys; sys.exit(0 if float($final_img_per_sec) > float($speed_threshold) else 1)"

}

export HOME=/workdir
cd /workdir && pip install . && git clone https://github.com/BaguaSys/examples.git
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
source $HOME/.cargo/env
pip install git+https://github.com/BaguaSys/bagua-core@master
algorithms=(gradient_allreduce)
for algotirhm in ${algorithms[@]}
do
    echo "begin to test ["${algorithms}]
    logfile=$(mktemp /tmp/bagua_benchmark_${algorithms}.XXXXXX.log)
    python -m bagua.distributed.launch \
        --nnodes=2 \
        --nproc_per_node 4 \
        --node_rank=0 \
        --master_addr="10.158.66.134" \
        --master_port=1234 \
        ${SYNTHETIC_SCRIPT} \
        --num-iters 100 \
        --algorithm gradient_allreduce \
        --deterministic \
        2>&1 | tee ${logfile}
    check_benchmark_log ${logfile}
done
