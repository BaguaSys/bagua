# Speed Benchmark Example

```bash
pip install -r requirements.txt
python3 -m bagua.distributed.launch \
    --nproc_per_node=8 \
    synthetic_benchmark.py \
        --algorithm allreduce
```
