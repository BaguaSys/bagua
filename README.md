<p align="center">
<img src="./figures/logo.png" width="200px"/>
</p>
<hr/>

[![tutorials](https://github.com/BaguaSys/tutorials/actions/workflows/tutorials.yml/badge.svg)](https://bagua-tutorials.kwai-seattle.com/) [![Documentation Status](https://readthedocs.org/projects/bagua/badge/?version=latest)](http://bagua.readthedocs.io/?badge=latest) [![Downloads](https://pepy.tech/badge/bagua/month)](https://pypi.org/project/bagua/) [![Docker Pulls](https://img.shields.io/docker/pulls/baguasys/bagua)](https://hub.docker.com/r/baguasys/bagua) [![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/baguasys/bagua)](https://hub.docker.com/r/baguasys/bagua) [![GitHub license](https://img.shields.io/github/license/BaguaSys/bagua)](https://github.com/BaguaSys/bagua/blob/master/LICENSE)

Bagua is a distributed training utility developed by [AI platform@Kuaishou Technology](https://www.kuaishou.com/en) and [DS3 Lab@ETH](https://ds3lab.inf.ethz.ch/). Users can extend the training on a single GPU to multi-GPUs (may across multiple machines) by simply adding a few lines of code. One prominent feature of Bagua is to provide a flexible system abstraction that supports state-of-the-art system relaxation techniques of distributed training. Powered by the new system design, Bagua has a great ability to implement and extend various state-of-the-art distributed learning algorithms. This in turns enables better scalability and efficiency of the end-to-end training process.
Researchers can also easily develop new distributed training algorithms within the Bagua framework, without worrying about low-level optimizations.

So far, Bagua has integrated communication primitives including

- Centralized Synchronous Communication (AllReduce)
- Decentralized Synchronous Communication
- Low Precision Communication

Its effectiveness has been evaluated in various scenarios, including VGG and ResNet on ImageNet, BERT Large and many industrial applications at Kuaishou.

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

For more comprehensive and up to date results, refer to [Bagua benchmark page](https://bagua-tutorials.kwai-seattle.com/benchmark/index.html).

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
  title={Distributed Learning Systems with First-Order Methods: An Instruction},
  author={Liu, J. and Zhang, C.},
  isbn={9781680837018},
  series={Foundations and trends in databases},
  url={https://books.google.com/books?id=vzQmzgEACAAJ},
  year={2020},
  publisher={now publishers}
}
```

## Limitations

* When communication is not a bottleneck in the training task, using Bagua communication algorithms will not provide significant performance improvement (unless you use other optimizations in Bagua such as fused optimizer).
* Currently only tested on Linux and NVIDIA GPUs.

## Links

* [Bagua Main Git Repo](https://github.com/BaguaSys/bagua)
* [Bagua Tutorials](https://bagua-tutorials.kwai-seattle.com/)
* [Bagua Examples](https://github.com/BaguaSys/examples)
* [Bagua API Documentation](https://bagua.readthedocs.io/)
