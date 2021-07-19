# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
==============================================
Qiskit Experiments (:mod:`qiskit_experiments`)
==============================================

.. currentmodule:: qiskit_experiments

*Note: This package is still under active development, there will be breaking
API changes, and re-organization of the package layout.*


Experiment Library
==================

See :mod:`qiskit_experiments.library` for a list of available experiments.

Experiment Data Classes
=======================

These container classes store the data and results from running experiments

.. autosummary::
    :toctree: ../stubs/

    ExperimentData


Experiment Base Classes
=======================

Construction of custom experiments should be done by making subclasses of the following
base classes

.. autosummary::
    :toctree: ../stubs/

    BaseExperiment
    BaseAnalysis
"""

from .version import __version__

# Base Classes
from .experiment_data import ExperimentData
from .base_analysis import BaseAnalysis
from .base_experiment import BaseExperiment

# Experiment modules
from . import library
from . import calibration_management
from . import data_processing
from . import database_service
from . import analysis

from . import composite
from . import characterization
from . import quantum_volume
