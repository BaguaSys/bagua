# Training imagenet using Bagua

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on ImageNet dataset.

## Training

### Single node, multiple GPUs

```bash
python3 -m bagua.distributed.launch --nproc_per_node=8 main.py --arch resnet50 --algorithm gradient_allreduce [imagenet-folder with train and val folders]
```

### Multiple nodes

The following scripts launch a distributed job on a 2-by-8 gpu cluster. 

Node 0:
```bash
python3 -m bagua.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=[master addr] --master_port [master port] main.py --arch resnet50 --algorithm gradient_allreduce [imagenet-folder with train and val folders]
```

Node 1:
```bash
python3 -m bagua.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=[master addr] --master_port [master port] main.py --arch resnet50 --algorithm gradient_allreduce [imagenet-folder with train and val folders]
```

## Usage

```
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N]
               [--warmup-epochs WARMUP_EPOCHS] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--wd W] [--milestones MILESTONES]
               [--gama GAMA] [-p N] [--resume PATH] [--save-checkpoint] [-e]
               [--pretrained] [--seed SEED] [--amp] [--prof PROF]
               [--algorithm ALGORITHM]
               DIR
```
