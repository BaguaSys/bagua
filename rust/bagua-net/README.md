Bagua-Net
=====

Bagua-Net is a high performance NCCL plugin for [Bagua](https://github.com/BaguaSys/bagua). By optimizing the fairness among multiple TCP streams, Bagua-Net can improve the overall throughput under TCP network by more than 30% for typical distributed learning tasks.

## Requirement

 - NCCL >= v2.6.4

## Quick Start

```bash
# Install Bagua-Net
cd cc && make
export BAGUA_NET_LIBRARY_PATH=$(readlink -f .)

# Install nccl and nccl-test
git clone https://github.com/NVIDIA/nccl.git && cd nccl && git checkout v2.10.3-1
make -j src.build && make install
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1

# Run nccl baseline
mpirun \
  --allow-run-as-root \
  -H ${HOST1}:1,${HOST2}:1 --np 2 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include eth01 \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# Run nccl with bagua-net
mpirun \
  --allow-run-as-root \
  -H ${HOST1}:1,${HOST2}:1 --np 2 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include eth01 \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BAGUA_NET_LIBRARY_PATH \
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
# If the installation is successful, there will be a log like this `NCCL INFO Using network BaguaNet`.
```

## Benchmark

On 4 nodes, each one equipped with 8 V100 GPUs and 100Gb ethernet connection, [the throughput of AllReduce can be improved by 50%](https://github.com/BaguaSys/bagua/wiki/NCCL-benchmark:-Bagua-Net-vs-google-fastsocket-vs-baseline).

In an [end-to-end test](https://github.com/BaguaSys/bagua/blob/master/examples/benchmark/synthetic_benchmark.py) with VGG16 as the model to be trained, with Bagua-Net enabled, Bagua can be more than 35% faster than other distributed solutions (such as PyTorch DDP):

```
# VGG16 on 4x8xV100 bagua-net
Running benchmark...
Iter #0: 4081.0 img/sec GPU
Iter #1: 4072.0 img/sec GPU
Iter #2: 4106.4 img/sec GPU
Iter #3: 4081.7 img/sec GPU
Iter #4: 4064.8 img/sec GPU
Iter #5: 4122.1 img/sec GPU
Iter #6: 3857.7 img/sec GPU
Iter #7: 4128.3 img/sec GPU
Iter #8: 4125.5 img/sec GPU
Iter #9: 3826.6 img/sec GPU
Img/sec per GPU: 126.5 +-6.4
Total img/sec on 32 GPU(s): 4046.6 +-205.2

# VGG16 on 4x8xV100 baseline
Running benchmark...
Iter #0: 2620.2 img/sec GPU
Iter #1: 2771.9 img/sec GPU
Iter #2: 2772.6 img/sec GPU
Iter #3: 2794.5 img/sec GPU
Iter #4: 2627.9 img/sec GPU
Iter #5: 2787.8 img/sec GPU
Iter #6: 2775.9 img/sec GPU
Iter #7: 2741.6 img/sec GPU
Iter #8: 2760.0 img/sec GPU
Iter #9: 2796.6 img/sec GPU
Img/sec per GPU: 85.8 +-3.8
Total img/sec on 32 GPU(s): 2744.9 +-122.3
```

## Limitations

Bagua-Net currently only accelerates AllReduce on TCP. Support for other operations and RDMA is still under development.
