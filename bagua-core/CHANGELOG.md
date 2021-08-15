# CHANGELOG

## [0.4.1] - 2021-08-14

    ### Features
    
        - Add opentelemetry to report tensor ready order (#42)


## [0.4.0] - 2021-07-23

    ### Features
    
        - Better debug log including tensor info when executing ops
        - Make full precision decentralized op stateless (#36)


## [0.3.1] - 2021-07-01

    ### Bug Fixes
    
        - Always mark bagua padding tensor as ready


## [0.3.0] - 2021-07-01

    ### Bug Fixes
    
        - Fix decompress incorrect pointer and typo in error msg
        - Fix python gil deadlock during getting data ptr

    ### Features
    
        - Replace NCCL with Aluminum (#7)
        - Support creating BaguaTensor by passing torch tensor directly (#19)
        - Compatible mode for getting pytorch tensor info with Python interpreter


## [0.2.0] - 2021-06-17

    ### Features
    
        - Initial support for python op (#2)


## [0.1.3] - 2021-06-17

    ### Bug Fixes
    
        - Move import bagua_install_library to install library function
        - Merge bagua_install_library and setup.py, remove nccl<=2.6 support


## [0.1.2] - 2021-06-17

    ### Features
    
        - Add version.py placeholder to prevent file not found error


## [0.1.1] - 2021-06-10

    ### Bug Fixes
    
        - Only run publish once on git tag

    ### Features
    
        - Install nccl deps in bagua core and add generated __version__ variable


## [0.1.0] - 2021-06-10

    ### Bug Fixes
    
        - Fix ci pypi versioning
        - Remove __init__.py and python __version__, use cargo version

    ### Features
    
        - Initial commit of bagua core impl
        - Add python packaging related files
        - Only publish pypi for master commits
        - Add __version__ variable


