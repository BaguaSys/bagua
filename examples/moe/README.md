Use the following script to start training locally with 8 gpus:

```bash
python3 -m bagua.distributed.launch --nproc_per_node=8 mnist_main.py --algorithm gradient_allreduce --num-local-experts 2
```
