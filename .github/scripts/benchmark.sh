#!/usr/bin/env bash

set -Eeuxo pipefail

function check_training_log {
    logfile=$1
    loss_limit=$2
    time_limit=$3

    cat $logfile
    final_loss=$(cat ${logfile} | grep "Test set: Average loss:" | tail -n 1 | awk '{print $5}' | awk -F',' '{print $1}')
    total_time=$(cat ${logfile} | grep "Total time used:" | tail -n 1 | awk '{print $NF}')

    if [ 0 -eq "$(echo "${final_loss} ${loss_limit}" | awk '{if ($1 == $2) print 1; else print 0}')" ]; then
        exit -1
    fi
#    if [ 0 -eq "$(echo "${total_time} ${time_limit}" | awk '{if ($1 < $2) print 1; else print 0}')" ]; then
#        exit -1
#    fi
}

function parse_training_log {
    logfile=$1    

    final_loss=$(cat ${logfile} | grep "Test set: Average loss:" | tail -n 1 | awk '{print $5}' | awk -F',' '{print $1}')
    total_time=$(cat ${logfile} | grep "Total time used:" | tail -n 1 | awk '{print $NF}')

    echo $final_loss $total_time
}

WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SHM_DISABLE=1

horovodrun -np 8  \
    python3 ${WORK_DIR}/examples/mnist/pytorch_mnist_hvd.py \
        --epochs 1 --deterministic \
    2>&1 | tee test.log
read hvd_loss hvd_time < <(parse_training_log test.log)
echo "horovod loss:" $hvd_loss "horovod time:" $hvd_time

horovodrun -np 8  \
    python3 ${WORK_DIR}/examples/mnist/pytorch_mnist_hvd.py \
        --epochs 1 --deterministic --amp \
    2>&1 | tee test.log
read hvd_amp_loss hvd_amp_time < <(parse_training_log test.log)
echo "horovod amp loss:" $hvd_amp_loss "horovod amp time:" $hvd_amp_time

python3 \
    -m bagua.distributed.launch \
    --nproc_per_node=8 \
    ${WORK_DIR}/examples/mnist/pytorch_mnist.py \
        --algorithm allreduce --epochs 1 --deterministic \
    2>&1 | tee test.log
check_training_log test.log 0.0475480623 20.0

python3 \
    -m bagua.distributed.launch \
    --nproc_per_node=8 \
    ${WORK_DIR}/examples/mnist/pytorch_mnist.py \
        --algorithm allreduce --epochs 1 --deterministic --amp \
    2>&1 | tee test.log
check_training_log test.log 0.0470389885 25.0

python3 \
    -m bagua.distributed.launch \
    --nproc_per_node=8 \
    ${WORK_DIR}/examples/mnist/pytorch_mnist.py \
        --algorithm quantize --epochs 1 --deterministic \
    2>&1 | tee > test.log
check_training_log test.log 0.0520482709 20.0


python3 \
    -m bagua.distributed.launch \
    --nproc_per_node=8 \
    ${WORK_DIR}/examples/mnist/pytorch_mnist.py \
        --algorithm decentralize --epochs 2 --deterministic \
    2>&1 | tee > test.log
check_training_log test.log 0.0552426365 30.0

python3 \
    -m bagua.distributed.launch \
    --nproc_per_node=8 \
    ${WORK_DIR}/examples/mnist/pytorch_mnist.py \
        --algorithm decentralize --epochs 2 --deterministic \
        --switch-to-algorithm allreduce --switch-epochs 1 \
    2>&1 | tee > test.log
check_training_log test.log 0.0426077106 30.0

python3 \
    -m bagua.distributed.launch \
    --nproc_per_node=8 \
    ${WORK_DIR}/examples/mnist/pytorch_mnist.py \
        --algorithm allreduce --epochs 1 --fuse-optimizer --deterministic \
    2>&1 | tee test.log
check_training_log test.log 0.0475480623 20.0

