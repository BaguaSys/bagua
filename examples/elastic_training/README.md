Use the following script to start elastic training with 1 ~ 4 node * 8 gpus:

Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

```bash
python3 -m bagua.distributed.run \
        --nnodes=1:4 \
        --nproc_per_node=8 \
        --rdzv_id=JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=192.168.1.1:1234 \
        main.py
```

You can dynamically add nodes 2 ~ 4 through the same command.
