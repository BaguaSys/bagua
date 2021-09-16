# Speed Benchmark Example

This directory contains the synthetic benchmark example. To run the example, use the following command (with 8 GPUs in this commands):

```bash
pip install -r requirements.txt
python3 -m bagua.distributed.launch \
    --nproc_per_node=8 \
    synthetic_benchmark.py \
        --algorithm gradient_allreduce
```
