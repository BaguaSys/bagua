Bagua
======
[![Generic badge](https://img.shields.io/badge/website-up-green.svg)](https://baguasys.github.io/tutorials/) [![Documentation Status](https://readthedocs.org/projects/bagua/badge/?version=latest)](http://bagua.readthedocs.io/?badge=latest) [![PyPI version](https://badge.fury.io/py/bagua.svg)](https://badge.fury.io/py/bagua) [![Docker](https://img.shields.io/badge/docker-pass-green)](https://hub.docker.com/r/baguasys/bagua) [![GitHub license](https://img.shields.io/github/license/BaguaSys/bagua)](https://github.com/BaguaSys/bagua/blob/master/LICENSE)

Bagua is a distributed training utility developed by [Kuaishou Technology](https://www.kuaishou.com/en) and [DS3 Lab@ETH](https://ds3lab.inf.ethz.ch/). Users can extend the training on a single GPU to multi-GPUs (may across multiple machines), with excellent speedup guarantee, by simply adding a few lines of code. Bagua also provides a flexible system abstraction that supports state-of-the-art system relaxation techniques of distributed training. Powered by the new system design, Bagua has a great ability to implement and extend various state-of-the-art distributed learning algorithms. Researchers can easily develop new distributed training algorithms based on bagua, without sacrificing system performance.

So far, Bagua has integrated primitives including

- Centralized Synchronous Communication (AllReduce)
- Decentralized Synchronous Communication
- Low Precision Communication

Its effectiveness has been verified in various scenarios, including VGG and ResNet on ImageNet, Bert Large and many industrial applications at Kuaishou.

The underlying communication execution engine is in [bagua-core](https://github.com/BaguaSys/bagua-core), a library written in Rust.

## Performance

<p align="center">
    <img src="https://baguasys.github.io/tutorials/benchmark/figures/scalability_vgg16.png" width="350"/>
</p>
<p align="center">
    The scalability of different systems on VGG16 with up to 128 GPUs.
</p>

<br/>
<br/>

<p align="center">
    <img src="https://baguasys.github.io/tutorials/benchmark/figures/tradeoff_network_bert-large-bandwidth.png" width="350"/><img src="https://baguasys.github.io/tutorials/benchmark/figures/tradeoff_network_bert-large-latency.png" width="330"/>
</p>
<p align="center">
    Epoch time of BERT-Large Finetune under different network conditions for different systems.
</p>

For more comprehensive and up to date results, refer to [Bagua benchmark page](https://baguasys.github.io/tutorials/benchmark/index.html).

## Installation

Develop version:

```
pip install git+https://github.com/BaguaSys/bagua.git
```

Release version:

```
pip install bagua
```

## Build API documentation locally

```
pip install -r docs/doc-requirements.txt
make html
```

## Links

* [Bagua Main Git Repo](https://github.com/BaguaSys/bagua)
* [Bagua Tutorials](https://baguasys.github.io/tutorials)
* [Bagua Examples](https://github.com/BaguaSys/examples)
* [Bagua API Documentation](https://bagua.readthedocs.io/)
