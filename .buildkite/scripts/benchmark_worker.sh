#!/usr/bin/env bash

echo "$BUILDKITE_PARALLEL_JOB"
echo "$BUILDKITE_PARALLEL_JOB_COUNT"
echo "$BUILDKITE_BUILD_ID"
echo "${MASTER_ADDR}:${MASTER_PORT}"

set -euox pipefail

# 0. install bagua
cp -a /upstream /workdir
export WORKDIR=/workdir && cd $WORKDIR && bash .buildkite/scripts/install_bagua.sh || exit 1
apt-get update && apt-get install -y iputils-ping
ping ${MASTER_ADDR} -c 10

nvidia-smi

# 1. test communication_primitives api
echo "begin to test [communication_primitives]"
COMMUNICATION_SCRIPT="${WORKDIR}/examples/communication_primitives/main.py"
NCCL_SOCKET_IFNAME=^docker,lo,veth python -m bagua.distributed.run \
    --nnodes=2 \
    --nproc_per_node 4 \
    --rdzv_id=${BUILDKITE_BUILD_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ${COMMUNICATION_SCRIPT}

# 2. benchmark test with all communication algorithms
SYNTHETIC_SCRIPT="${WORKDIR}/examples/benchmark/synthetic_benchmark.py"
algorithms=(gradient_allreduce bytegrad decentralized low_precision_decentralized async qadam)
length=${#algorithms[@]}
for ((i = 0; i < $length; i++)); do
    echo "begin to test ["${algorithms[$i]}]
    logfile=$(mktemp /tmp/bagua_benchmark_${algorithms[$i]}.XXXXXX.log)
    NCCL_SOCKET_IFNAME=^docker,lo,veth GLOO_SOCKET_IFNAME=enp96s0f0 python -m bagua.distributed.run \
        --nnodes=2 \
        --nproc_per_node 4 \
        --rdzv_id=${BUILDKITE_BUILD_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        ${SYNTHETIC_SCRIPT} \
        --num-iters 100 \
        --algorithm ${algorithms[$i]} \
        --deterministic \
        --async-sync-interval 100 \
        --async-warmup-steps 100 \
        2>&1 | tee ${logfile}
done

# 3. test moe
MOE_SCRIPT="${WORKDIR}/examples/moe/mnist_main.py"
logfile=$(mktemp /tmp/bagua_moe_gradient_allreduce.XXXXXX.log)
NCCL_SOCKET_IFNAME=^docker,lo,veth CUDA_VISIBLE_DEVICES=0,1 python -m bagua.distributed.run \
    --nnodes=2 \
    --nproc_per_node 2 \
    --rdzv_id=${BUILDKITE_BUILD_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ${MOE_SCRIPT} \
    --algorithm gradient_allreduce \
    --epochs 5 \
    --num-local-experts 2 \
    --set-deterministic \
    2>&1 | tee ${logfile}
