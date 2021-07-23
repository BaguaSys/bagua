## v0.4.0

### ⚠ BREAKING CHANGE

* `BaguaBucketPy::append_decentralized_synchronous_op` now only supports full precision decentralized communication.

### Features

* make full precision decentralized op stateless (#36) 98319c9
* add (scatter, gather, scatter_reduce) and all inplace version communication primitives (#37) f931473
* add native low precision decentralized operator (#26) 50295e8
* better debug log including tensor info when executing ops 1bd6e0b

### Fixes

* substract overflow error for decentralized op (#39) 30cdb67
* fix scatter and reduce_scatter implementation (#40) ee90376
* make compress/decompress of BaguaTensor `method` string consistent (#33) ee929df

## v0.3.1

### Fixes

* always mark bagua padding tensor as ready 63f88d4
* fix alltoall_v parameter types (#27) b541d85

## v0.3.0

### ⚠ BREAKING CHANGE

* `BaguaBucketPy` and `BaguaTensorPy` now require name. `BaguaTensorPy` is created by passing pytorch tensor directly now.

### Features

* Compatible mode for getting pytorch tensor info with Python interpreter 1534d23
* Support creating BaguaTensor by passing torch tensor directly (#19) 4306e94
* Support Reduction op selection (SUM, MAX, etc.) b1bf784
* Add `reduce` and `allgather` python interface ff68a61
* Support `alltoall_v` (vector alltoall) (#14) a6fe110

### Fixes

* fix python gil deadlock during getting data ptr 6ba6ace
* fix decompress incorrect pointer and typo in error msg a7e34ba
