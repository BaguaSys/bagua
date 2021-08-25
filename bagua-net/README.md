`bagua-net`
=====

[![GitHub license](https://img.shields.io/github/license/BaguaSys/bagua-core)](https://github.com/BaguaSys/bagua-core/blob/master/LICENSE)

Bagua-Net is high performance NCCL plugin for Bagua. By optimizing the fairness of TCP multi-streams, it has achieved better performance under TCP network.

## QuickStart

```bash
# Install Bagua-Net
cargo build
cd cc && make
export BAGUA_NET_LIBRARY_PATH=$(readlink -f .):$(readlink -f ../target/debug)

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

We tested the performance on the Kuaishou V100 machine and found that the throughput of Bagua-Net allreduce was significantly better than the NCCL tcp. Here is the [test record](https://gist.github.com/shjwudp/9eb08bbaf46d9b12239aa62a54d35a48).

In the end-to-end test, we have obvious advantages compare to NCCL TCP. You can reproduce this test result through [this script](https://github.com/BaguaSys/examples/blob/main/benchmark/synthetic_benchmark.py).

```
# VGG16 on 4x8xV100 bagua-net
Running benchmark...
Iter #0: 3643.4 img/sec GPU
Iter #1: 3648.4 img/sec GPU
Iter #2: 3544.0 img/sec GPU
Iter #3: 3656.5 img/sec GPU
Iter #4: 3684.8 img/sec GPU
Iter #5: 3641.1 img/sec GPU
Iter #6: 3643.4 img/sec GPU
Iter #7: 3590.5 img/sec GPU
Iter #8: 3635.0 img/sec GPU
Iter #9: 3694.8 img/sec GPU
Img/sec per GPU: 113.7 +-2.5
Total img/sec on 32 GPU(s): 3638.2 +-80.9

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
