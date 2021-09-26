# CHANGELOG

## [0.8.0] - 2021-09-26

### Bug Fixes

#### Ci

- Only run publish once on git tag

#### Core

- Fix compressed buffer can not be scattered to odd number of ranks

#### Other

- Fix ci pypi versioning
- Remove __init__.py and python __version__, use cargo version
- Move import bagua_install_library to install library function
- Merge bagua_install_library and setup.py, remove nccl<=2.6 support
- Fix alltoall_v parameter (#17)
- Reduce and allgather python interface
- Fix decompress incorrect pointer and typo in error msg
- Fix python gil deadlock during getting data ptr
- Fix benchmark script requirements
- Fix alltoall_v parameter types (#27)
- Always mark bagua padding tensor as ready
- Make compress/decompress of BaguaTensor `method` string consistent (#33)
- Fix scatter and reduce_scatter implementation (#40)
- Substract overflow error for decentralized op (#39)
- Fix QADAM params (#17)
- Fix assert precision (#18)
- Replace mutex with atomic bool for async op and add Aluminum submodule update (#67)
- Fix duplicated dependency downloading during installation (#77)
- Fix async algorithm aborting and hanging (#78, #81)
- Fix qadam algorithm call (#20)
- Fix missing symbols in the zip library (#24)
- Fix random autotune server hang (#206)
- Bagua-net library path mismatch, make `--enable_bagua_net` argument style consistent with other args (#218)

#### Python

- Fix random autotune-service hang
- Handle conflicts caused by sklearn upgrade (#225)

### Features

#### Ci

- Only publish pypi for master commits

#### Other

- Add async model average algorithm (#110)
- Add cached dataset wrapper (#148)
- Support sync batchnorm (#151)
- Add `--enable-bagua-net` option in launcher (#183)
- Add pytorch examples for MNIST, ImageNet, SQuAD training (#1)
- Add requirements.txt, only download dataset on local rank 0 (#2)
- Add python packaging related files
- Add `__version__` variable
- Install nccl deps in bagua core and add generated `__version__` variable
- Add version.py placeholder to prevent file not found error
- Initial support for python op (#2)
- Add 5 min timeout for buckets' comm op (#5)
- Replace NCCL with Aluminum (#7)
- Add synethetic benchmark script (#5)
- Add elastic training example (#7)
- Support alltoall_v (vector alltoall) (#14)
- Add reduce and allgather python interface
- Support reduce and allgather op with Reduction op enum
- Support creating BaguaTensor by passing torch tensor directly (#19)
- Compatible mode for getting pytorch tensor info with Python interpreter
- Better debug log including tensor info when executing ops
- Add native low precision decentralized operator (#26)
- Add (scatter, gather, scatter_reduce) and all inplace version communication primitives (#37)
- Make full precision decentralized op stateless (#36)
- Add communication_primitives example (#12)
- Use nccl 2.10 avg op for all algorithms using averaging (#46, #45)
- Add opentelemetry to report tensor ready order (#42)
- Add deterministic flag (#15)
- Add native async model average algorithm (#41)
- Add examples for async model average algorithm (#14)
- Support packet splitting and multi-stream parallel transmission (#5)
- Support ncclnet v3 and remove the dependency on nccl in the installation environment (#17)
- Add sync interval param to async examples (#19)
- Suppport tokio backend (#21)
- Support bagua-net (#89)

#### Python

- Broadcast scalars for optimizers (#202)

## [0.7.0] - 2021-08-16

### Bug Fixes

- Make compress/decompress of BaguaTensor `method` string consistent (#33)
- Fix scatter and reduce_scatter implementation (#40)
- Substract overflow error for decentralized op (#39)
- Autotune api conflict (#131)
- Autotune pytest run forever (#132)
- Fix bagua.distributed.run --is_output_autotune_log parsing (#145)
- Fix QADAM params (#17)
- Fix assert precision (#18)
- Fix torch version check (#150)

### Features

- Add native low precision decentralized operator (#26)
- Add low precision decentralized algorithm (#103)
- Add (scatter, gather, scatter_reduce) and all inplace version communication primitives (#37)
- Add all communication primitives such as send recv to communication module (#128)
- Make full precision decentralized op stateless (#126)
- Make full precision decentralized op stateless (#36)
- Add communication_primitives example (#12)
- Support duplicated parameters acorss different modules (#147)
- Support nccl 2.10 ReduceOp.AVG (#149)
- Support nccl 2.10 ncclAvg (#45)
- Use nccl 2.10 avg op for all algorithms using averaging (#46)
- Add opentelemetry to report tensor ready order (#42)
- Add support for reporting tensor completion order (#146)
- Add deterministic flag (#15)


## [0.6.3] - 2021-07-08

### Bug Fixes

- Autotune service defaults with a fixed random seed (#117)

### Features

- Improve autotune speed metrics measurement for better accuracy (#86)
- Install.sh will not install rust if already exist on the system
- Install.sh upgrades existing bagua
- Sort q_adam variables for better performance (#102)
- Better debug log including tensor info when executing ops
- Support multiple models on autotune service (#107)
- Support multiple models in buckets registration (#113)
- Support different ssh port on different nodes (#93)


## [0.6.2] - 2021-07-02

### Bug Fixes

- Fix QAdam gradient is not BaguaTensor during first stage


## [0.6.1] - 2021-07-02

### Bug Fixes

- Fix alltoall_v parameter types (#27)
- Fix BaguaBacket.clear_ops() return value
- Always mark bagua padding tensor as ready
- Fix append python op callable reference
- BaguaBucket.tensors should only contain original passed in tensors

### Features

- Add append op methods to python `BaguaBucket` class (#87)
- Wrap python op in communication stream context by default
- Broadcast model parameters on every algorithm reset
- Add QAdam algorithm (#92)


## [0.6.0] - 2021-07-01

### Bug Fixes

- The environment variable LOCAL_SIZE has been renamed in LOCAL_WORLD_SIZE (#51)
- Fix alltoall_v parameter (#17)
- Reduce and allgather python interface
- Fix decompress incorrect pointer and typo in error msg
- Fix python gil deadlock during getting data ptr
- Auto installation for centos (#66)
- Fix algoirthm pre forward hook not returned
- Fix benchmark script requirements

### Features

- Add synethetic benchmark script (#5)
- Auto installation support centos (#50)
- Add elastic training example (#7)
- Support alltoall_v (vector alltoall) (#14)
- Add reduce and allgather python interface
- Support reduce and allgather op with Reduction op enum
- Support reduction op and reduce
- Support creating BaguaTensor by passing torch tensor directly (#19)
- Compatible mode for getting pytorch tensor info with Python interpreter
- Add algorithm import in bagua.torch_api
- Add all algorithms import in bagua.torch_api.algorithms


## [0.5.0] - 2021-06-25

### Bug Fixes

- Do not setup python dependencies when performing codeql check
- Remove logging in load balancing dataloader to avoid deadlock (#35)
- Add back user interfacing imports in init.py (#38)
- Fix bucket size switch not effective (#48)

### Features

- Add broadcast_buffer in bagua_init (#29)
- Elastic training (#31)
- Add 5 min timeout for buckets' comm op (#5)
- Replace NCCL with Aluminum (#7)
- Add dependency installation script for ubuntu (#41)


## [0.4.0] - 2021-06-17

### Bug Fixes

- Fix ci pypi versioning
- Remove __init__.py and python __version__, use cargo version
- Only run publish once on git tag
- Fix baguaelastic launcher
- Fix baguaelastic launch script
- Fix setup.py for low version setuptools (#14)
- Move import bagua_install_library to install library function
- Merge bagua_install_library and setup.py, remove nccl<=2.6 support

### Features

- Add pytorch examples for MNIST, ImageNet, SQuAD training (#1)
- Add requirements.txt, only download dataset on local rank 0 (#2)
- Initial commit of bagua core impl
- Add python packaging related files
- Only publish pypi for master commits
- Add __version__ variable
- Install nccl deps in bagua core and add generated __version__ variable
- Initial public release of bagua python code
- Update interface and doc for loadbalance dataloader and add doc for fused optimizer (#17)
- Add version.py placeholder to prevent file not found error
- Initial support for python op (#2)
- Support new python op supported backend (#26)


