# CHANGELOG

## [0.7.0] - 2021-08-15

    ### Bug Fixes
    
        - Autotune api conflict (#131)

    ### Features
    
        - Add low precision decentralized algorithm (#103)
        - Add all communication primitives such as send recv to communication module (#128)
        - Make full precision decentralized op stateless (#126)
        - Support nccl 2.10 ReduceOp.AVG (#149)
        - Add support for reporting tensor completion order (#146)


## [0.6.3] - 2021-07-08

    ### Features
    
        - Install.sh will not install rust if already exist on the system
        - Install.sh upgrades existing bagua
        - Sort q_adam variables for better performance (#102)
        - Support multiple models on autotune service (#107)
        - Support multiple models in buckets registration (#113)
        - Support different ssh port on different nodes (#93)


## [0.6.2] - 2021-07-02

    ### Bug Fixes
    
        - Fix QAdam gradient is not BaguaTensor during first stage


## [0.6.1] - 2021-07-02

    ### Bug Fixes
    
        - Fix BaguaBacket.clear_ops() return value
        - Fix append python op callable reference
        - BaguaBucket.tensors should only contain original passed in tensors

    ### Features
    
        - Wrap python op in communication stream context by default
        - Broadcast model parameters on every algorithm reset
        - Add QAdam algorithm (#92)


## [0.6.0] - 2021-07-01

    ### Bug Fixes
    
        - Fix algoirthm pre forward hook not returned

    ### Features
    
        - Support reduction op and reduce
        - Add algorithm import in bagua.torch_api
        - Add all algorithms import in bagua.torch_api.algorithms


## [0.5.0] - 2021-06-25

    ### Bug Fixes
    
        - Do not setup python dependencies when performing codeql check
        - Remove logging in load balancing dataloader to avoid deadlock (#35)

    ### Features
    
        - Add broadcast_buffer in bagua_init (#29)
        - Elastic training (#31)
        - Add dependency installation script for ubuntu (#41)


## [0.4.0] - 2021-06-17

    ### Bug Fixes
    
        - Fix baguaelastic launcher
        - Fix baguaelastic launch script

    ### Features
    
        - Initial public release of bagua python code


