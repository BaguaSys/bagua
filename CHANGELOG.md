## v0.6.2

### Fixes

* fix QAdam gradient is not BaguaTensor during first stage 1d4dc82

## v0.6.1

### Features

* add QAdam algorithm (#92) 0dafd24
* broadcast model parameters on every algorithm reset e5b36dc
* wrap python op in communication stream context by default 51eb656
* add append op methods to python `BaguaBucket` class (#87) 84d8cbc

### Fixes

* BaguaBucket.tensors should only contain original passed in tensors c4ff05f
* fix append python op callable reference 04019cc
* fix BaguaBacket.clear_ops() return value 8cb9f54

## v0.6.0

### ⚠ BREAKING CHANGE

* Now end users should use `model.with_bagua(...)` API to use Bagua for communication. Algorithm developers can use `bagua.torch_api.algorithms.Algorithm` to easily develop new algorithms. Installation requires `bagua-core` >=0.3 now.

### Features

* add algorithm import in bagua.torch_api ee73edc
* support reduction op and reduce ac8632c
* auto installation support centos (#50) 073a59e

### Fixes

* fix algoirthm pre forward hook not returned e6c7c8d
* the environment variable LOCAL_SIZE has been renamed in LOCAL_WORLD_SIZE (#51) 801b25a
