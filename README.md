<p align="center">
<img src="https://user-images.githubusercontent.com/18649508/136457975-e4d81ced-0e43-4793-8865-1379e82921f9.png" width="200px"/>
</p>

<hr/>

<div align="center">
<a href="https://bagua-tutorials.kwai-seattle.com/"><img src="https://img.shields.io/badge/tutorials-passing-green" alt="tutorials"></a> <a href="http://bagua.readthedocs.io/?badge=latest"><img src="https://readthedocs.org/projects/bagua/badge/?version=latest" alt="Documentation Status"></a> <a href="https://pypi.org/project/bagua/"><img src="https://pepy.tech/badge/bagua/month" alt="Downloads"></a> <a href="https://hub.docker.com/r/baguasys/bagua"><img src="https://img.shields.io/docker/pulls/baguasys/bagua" alt="Docker Pulls"></a> <a href="https://hub.docker.com/r/baguasys/bagua"><img src="https://img.shields.io/docker/cloud/build/baguasys/bagua" alt="Docker Cloud Build Status"></a> <a href="https://github.com/BaguaSys/bagua/blob/master/LICENSE"><img src="https://img.shields.io/github/license/BaguaSys/bagua" alt="GitHub license"></a>
</div>

<br/>

Bagua is a deep learning training acceleration framework for PyTorch developed by [AI platform@Kuaishou Technology](https://www.kuaishou.com/en) and [DS3 Lab@ETH](https://ds3lab.inf.ethz.ch/). Bagua currently supports:

- **Advanced Distributed Training Algorithms**: Users can extend the training on a single GPU to multi-GPUs (may across multiple machines) by simply adding a few lines of code (optionally in [elastic mode](https://bagua-tutorials.kwai-seattle.com/elastic-training/)). One prominent feature of Bagua is to provide a flexible system abstraction that supports state-of-the-art system relaxation techniques of distributed training. So far, Bagua has integrated communication primitives including
  - Centralized Synchronous Communication (e.g. [Gradient AllReduce](https://bagua-tutorials.kwai-seattle.com/algorithms/gradient-allreduce))
  - Decentralized Synchronous Communication (e.g. [Decentralized SGD](https://bagua-tutorials.kwai-seattle.com/algorithms/decentralized))
  - Low Precision Communication (e.g. [ByteGrad](https://bagua-tutorials.kwai-seattle.com/algorithms/bytegrad))
  - Asynchronous Communication (e.g. [Async Model Average](https://bagua-tutorials.kwai-seattle.com/algorithms/async-model-average))
- [**TCP Communication Acceleration (Bagua-Net)**](https://bagua-tutorials.kwai-seattle.com/more-optimizations/bagua-net): Bagua-Net is a low level communication acceleration feature provided by Bagua. It can greatly improve the throughput of AllReduce on TCP network. You can enable Bagua-Net optimization on any distributed training job that uses NCCL to do GPU communication (this includes PyTorch-DDP, Horovod, DeepSpeed, and more).
- [**Performance Autotuning**](https://bagua-tutorials.kwai-seattle.com/performance-autotuning/): Bagua can automatically tune system parameters to achieve the highest throughput.
- [**Generic Fused Optimizer**](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/fused_optimizer/index.html): Bagua provides generic fused optimizer which improve the performance of optimizers by fusing the optimizer `.step()` operation on multiple layers. It can be applied to arbitrary PyTorch optimizer, in contrast to [NVIDIA Apex](https://nvidia.github.io/apex/optimizers.html)'s approach, where only some specific optimizers are implemented.
- [**Load Balanced Data Loader**](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/contrib/load_balancing_data_loader/index.html): When the computation complexity of samples in training data are different, for example in NLP and speech tasks, where each sample have different lengths, distributed training throughput can be greatly improved by using Bagua's load balanced data loader, which distributes samples in a way that each worker's workload are similar.

Its effectiveness has been evaluated in various scenarios, including VGG and ResNet on ImageNet, BERT Large and many industrial applications at Kuaishou.

## Links

* [Bagua Main Git Repo](https://github.com/BaguaSys/bagua)
* [Bagua Tutorials](https://bagua-tutorials.kwai-seattle.com/)
* [Bagua Examples](https://github.com/BaguaSys/bagua/tree/master/examples)
* [Bagua API Documentation](https://bagua.readthedocs.io/)

## Performance

<p align="center">
    <img src="https://bagua-tutorials.kwai-seattle.com/benchmark/figures/e2e_vgg16_128.png" width="600"/>
</p>
<p align="center">
    The performance of different systems and algorithms on VGG16 with 128 GPUs under different network bandwidth.
</p>

<br/>
<br/>

<p align="center">
    <img src="https://bagua-tutorials.kwai-seattle.com/benchmark/figures/tradeoff_network_bert-large-bandwidth.png" width="250"/><img src="https://bagua-tutorials.kwai-seattle.com/benchmark/figures/tradeoff_network_bert-large-latency.png" width="250"/><img src="https://bagua-tutorials.kwai-seattle.com/benchmark/figures/tradeoff_network_legend.png" width="260"/>
</p>
<p align="center">
    Epoch time of BERT-Large Finetune under different network conditions for different systems.
</p>

For more comprehensive and up to date results, refer to [Bagua benchmark page](https://bagua-tutorials.kwai-seattle.com/benchmark/index.html).

## Installation

Wheels (precompiled binary packages) are available for Linux (x86_64). Package names are different depending on your CUDA Toolkit version (CUDA Toolkit version is shown in `nvcc --version`).

| CUDA Toolkit version | Installation command      |
|----------------------|---------------------------|
| >= v10.2             | pip install bagua-cuda102 |
| >= v11.1             | pip install bagua-cuda111 |
| >= v11.3             | pip install bagua-cuda113 |

Add `--pre` to `pip install` commands to install pre-release (development) versions. See [Bagua tutorials](https://bagua-tutorials.kwai-seattle.com/getting-started/) for quick start guide and more installation options.

## Quick Start on AWS

Thanks to the [Amazon Machine Images (AMI)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html), we can provide users an easy way to deploy and run Bagua on AWS EC2 clusters with flexible size of machines and a wide range of GPU types. Users can find our pre-installed Bagua image on EC2 by a unique AMI-ID that we publish here. 

| Bagua version  | AMI ID |  Region |
|---|---|---|
| 0.6.3 | ami-0e719d0e3e42b397e | us-east-1 |

To manage the EC2 cluster more efficiently, we use [Starcluster](http://star.mit.edu/cluster/) as a toolkit to manipulate the cluster. In the `config` file of Starcluster, there are a few configurations that need to be set up by users, including AWS credentials, cluster settings, etc. More information regarding the Starcluster configuration can be found in this [tutorial](http://star.mit.edu/cluster/docs/latest/quickstart.html). Note that AMI is a regional resource, so you need to specify the AMI ID and its corresponding EC2 region at the same time.

For example, we create a EC2 cluster with 4 machines (`p3.16xlarge`), each of which has 8 V100 GPUs. The cluster is based on the Bagua AMI we pre-installed in `us-east-1` region. Then the `config` file of Starcluster would be:

```yaml
# region of EC2 instances, here we choose us_east_1
AWS_REGION_NAME = us-east-1
AWS_REGION_HOST = ec2.us-east-1.amazonaws.com
# AMI ID of Bagua
NODE_IMAGE_ID = ami-0e719d0e3e42b397e
# number of instances
CLUSTER_SIZE = 4
# instance type
NODE_INSTANCE_TYPE = p3.16xlarge
```

With above setup, we created two identical clusters to benchmark a synthesized image classification task over Bagua and Horovod, respectively. Here is the screen recording video of this experiment. 

<p align="center">
    <a href="https://youtu.be/G8o5HVYZJvs"><img src="https://user-images.githubusercontent.com/18649508/136463585-ba911d20-9088-48b7-ab32-fc3e465c31b8.png" width="600"/></a>
</p>

## Cite Bagua

```bibtex
% System Overview
@misc{gan2021bagua,
  title={BAGUA: Scaling up Distributed Learning with System Relaxations}, 
  author={Shaoduo Gan and Xiangru Lian and Rui Wang and Jianbin Chang and Chengjun Liu and Hongmei Shi and Shengzhuo Zhang and Xianghong Li and Tengxu Sun and Jiawei Jiang and Binhang Yuan and Sen Yang and Ji Liu and Ce Zhang},
  year={2021},
  eprint={2107.01499},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

% Theory on System Relaxation Techniques
@book{liu2020distributed,
  title={Distributed Learning Systems with First-Order Methods: An Introduction},
  author={Liu, J. and Zhang, C.},
  isbn={9781680837018},
  series={Foundations and trends in databases},
  url={https://books.google.com/books?id=vzQmzgEACAAJ},
  year={2020},
  publisher={now publishers}
}
```

## Discussion

Feel free to join our [Zulip chat](https://bagua.zulipchat.com) for discussion!

You can also scan the following QR code to join our WeChat group :)

<img src="https://user-images.githubusercontent.com/18649508/137250428-c9845337-baf0-4dca-b8b4-e2bbe97250f4.png" width="300"/>
