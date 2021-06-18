"""

Bagua_ is a communication library
developed by Kuaishou Technology and DS3 Lab for deep learning.

See tutorials_ for Bagua's rationale and benchmark.

.. _Bagua: https://github.com/BaguaSys/bagua
.. _tutorials: https://baguasys.github.io/tutorials/
"""

import bagua_core
from . import torch_api, autotune, script, bagua_define
from .version import __version__
