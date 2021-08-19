#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"

set -euox pipefail

CHECK_RESULT=()
function check_benchmark_log {
    logfile=$1
    algorithm=$2
    speed=$3
    loss=$4

    final_batch_loss=$(cat ${logfile} | grep "TrainLoss" | tail -n 1 | awk '{print $4}')
    img_per_sec=$(cat ${logfile} | grep "Img/sec per " | tail -n 1 | awk '{print $4}')

    echo "Checking ["${algorithm}"]..."
    if [ $final_batch_loss == $loss ]; then
        echo "Check ["${algorithm}"] success, final_batch_loss is equal."
    else
        result="Check ["${algorithm}"] fail, final_batch_loss["$final_batch_loss"] is not equal with "$loss"."
        echo $result
        CHECK_RESULT[${#CHECK_RESULT[*]}]="${result}\n"
    fi
    var=$(awk 'BEGIN{ print "'$img_per_sec'"<"'$speed'" }')
    if [ "$var" -eq 1 ]; then
        result="Check ["${algorithm}"] fail, img_per_sec["$img_per_sec"] is smaller than "$speed
        echo $result
        CHECK_RESULT[${#CHECK_RESULT[*]}]="${result}\n"
    else
        echo "Check ["${algorithm}"] success, img_per_secc["$img_per_sec"] is greater than "$speed
    fi
}

export HOME=/workdir
cd /workdir && pip install . && git clone https://github.com/BaguaSys/examples.git
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
source $HOME/.cargo/env
pip install git+https://github.com/BaguaSys/bagua-core@master

echo "begin to test [communication_primitives]"
COMMUNICATION_SCRIPT="/workdir/examples/communication_primitives/main.py"
python -m bagua.distributed.launch \
    --nnodes=2 \
    --nproc_per_node 4 \
    --node_rank=0 \
    --master_addr="10.158.66.134" \
    --master_port=1234 \
    ${COMMUNICATION_SCRIPT}

SYNTHETIC_SCRIPT="/workdir/examples/benchmark/synthetic_benchmark.py"
algorithms=(gradient_allreduce bytegrad decentralized low_precision_decentralized qadam)
speeds=(185.0 180.0 150.0 115.0 170)
losses=(0.001763 0.001694 0.002583 0.001821 0.000010)
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

if [ ${#CHECK_RESULT[*]} -gt 0 ]; then
  echo -e ${CHECK_RESULT[*]}
  exit 1
fi
