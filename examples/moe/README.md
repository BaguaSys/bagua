Use the following script to start training locally with 2 gpus:

```bash
python3 -m bagua.distributed.launch --nproc_per_node=2 main.py --algorithm gradient_allreduce
```
