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
