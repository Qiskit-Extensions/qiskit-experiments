# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===============================================================================================
Driven Frequency Tuning (:mod:`qiskit_experiments.library.driven_freq_tuning`)
===============================================================================================

.. currentmodule:: qiskit_experiments.library.driven_freq_tuning

Experiments
===========
.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    StarkRamseyXY
    StarkRamseyXYAmpScan
    StarkP1Spectroscopy


Analysis
========

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    StarkRamseyXYAmpScanAnalysis
    StarkP1SpectAnalysis


Utilities
=========

.. autosummary::
    :toctree: ../stubs/

    StarkCoefficients
    convert_amp_to_freq
    convert_freq_to_amp
    retrieve_coefficients_from_backend
    retrieve_coefficients_from_service
"""

from .analyses import StarkRamseyXYAmpScanAnalysis, StarkP1SpectAnalysis
from .ramsey import StarkRamseyXY
from .ramsey_amp_scan import StarkRamseyXYAmpScan
from .p1_spect import StarkP1Spectroscopy

from .coefficient_utils import (
    StarkCoefficients,
    convert_amp_to_freq,
    convert_freq_to_amp,
    retrieve_coefficients_from_backend,
    retrieve_coefficients_from_service,
)
