# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
Hamiltonian Characterization Experiments (:mod:`qiskit_experiments.library.hamiltonian`)
===============================================================================================

.. currentmodule:: qiskit_experiments.library.hamiltonian

This module provides a set of experiments to characterize qubit Hamiltonians.

HEAT Experiments
================

HEAT stands for `Hamiltonian Error Amplifying Tomography` which amplifies the
dynamics of an entangling gate along a specified axis of the target qubit. Here,
errors are typically amplified by repeating a sequence of gates which results in
a ping-pong pattern when measuring the qubit population.

HEAT for ZX Hamiltonian
-----------------------

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ZXHeat
    ZX90HeatXError
    ZX90HeatYError
    ZX90HeatZError

HEAT Analysis
-------------

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    HeatElementAnalysis
    HeatAnalysis

HEAT Base Classes
-----------------

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    HeatElement


"""

from .heat_base import HeatElement
from .heat_zx import ZXHeat, ZX90HeatXError, ZX90HeatYError, ZX90HeatZError
from .heat_analysis import HeatElementAnalysis, HeatAnalysis
