"""
cht_hurrywave
=============

Python interface to the HurryWave spectral wave model.

Exports the main :class:`HurryWave` domain object and the BMI/XMI wrapper
:class:`HurryWaveXmi`.
"""

version = "1.0.0"

from .hurrywave import HurryWave  # noqa: F401
from .xmi import HurryWaveXmi  # noqa: F401
