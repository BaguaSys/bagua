#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

SYNTHETIC_SCRIPT="/workdir/examples/benchmark/synthetic_benchmark.py"

function check_benchmark_log {
    logfile=$1
    speed=$3
    loss=$4

    final_batch_loss=$(cat ${logfile} | grep "TrainLoss" | tail -n 1 | awk '{print $4}')
    img_per_sec=$(cat ${logfile} | grep "Img/sec per " | tail -n 1 | awk '{print $4}')

    python -c "import sys; sys.exit(1) if float($final_batch_loss) != float($loss) else print('final_batch_loss is euqal.')"
    python -c "import sys; sys.exit(1) if float($img_per_sec) < float($speed) else print('imag_per_sec is bigger than $speed.')"
}

export HOME=/workdir
cd /workdir && pip install . && git clone https://github.com/BaguaSys/examples.git
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
source $HOME/.cargo/env
pip install git+https://github.com/BaguaSys/bagua-core@master

algorithms=(gradient_allreduce bytegrad decentralized low_precision_decentralized qadam)
speeds=(200.0 180.0 150.0 115.0 100)
losses=(0.001848 0.001815 0.002699 0.002047 0)
length=${#algorithms[@]}
for ((i=0;i<$length;i++))
do
    echo "begin to test ["${algorithms[$i]}]
    logfile=$(mktemp /tmp/bagua_benchmark_${algorithms[$i]}.XXXXXX.log)
    python -m bagua.distributed.launch \
        --nnodes=2 \
        --nproc_per_node 4 \
        --node_rank=0 \
        --master_addr="10.158.66.134" \
        --master_port=1234 \
        ${SYNTHETIC_SCRIPT} \
        --num-iters 100 \
        --algorithm ${algorithms[$i]} \
        --deterministic \
        2>&1 | tee ${logfile}
    check_benchmark_log ${logfile} ${algorithms[$i]} ${speeds[$i]} ${losses[$i]}
done

echo "begin to test [communication_primitives]"
COMMUNICATION_SCRIPT="/workdir/examples/communication_primitives/main.py"
python -m bagua.distributed.launch \
    --nnodes=2 \
    --nproc_per_node 4 \
    --node_rank=0 \
    --master_addr="10.158.66.134" \
    --master_port=1234 \
    ${COMMUNICATION_SCRIPT}
