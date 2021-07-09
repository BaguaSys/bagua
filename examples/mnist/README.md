Use the following script to start training locally with 8 gpus:

```bash
python3 -m bagua.distributed.launch --nproc_per_node=8 main.py --algorithm gradient_allreduce
```
