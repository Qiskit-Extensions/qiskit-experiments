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
=======================================================================
Calibration Experiments (:mod:`qiskit_experiments.library.calibration`)
=======================================================================

.. currentmodule:: qiskit_experiments.library.calibration

Calibrating qubit setups is the task of finding the pulse shapes and parameter
values that maximizes the fidelity of the resulting quantum operations. This
therefore requires experiments which are analyzed to extract parameter values.
Furthermore, the resulting parameter values and schedules must be managed. The
calibration module in Qiskit experiments allows users to run calibration
experiments and manage the resulting schedules and parameter values.

The following experiments are designed to calibrate parameter values. Some experiments such
as :class:`.QubitSpectroscopy` can both be seen as characterization and calibrations
experiments. Such experiments can be found in the
:mod:`~qiskit_experiments.library.characterization`
module.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    RoughFrequencyCal
    FrequencyCal
    FineFrequencyCal
    RoughDragCal
    FineDragCal
    FineXDragCal
    FineSXDragCal
    FineAmplitudeCal
    FineXAmplitudeCal
    FineSXAmplitudeCal
    HalfAngleCal
    RoughAmplitudeCal
    RoughXSXAmplitudeCal
    EFRoughXSXAmplitudeCal
    CrossResRoughAmplitudeCal

Calibrations management
=======================

See :mod:`.calibration_management`.
"""

from .rough_frequency import RoughFrequencyCal, RoughEFFrequencyCal
from .rough_drag_cal import RoughDragCal
from .rough_amplitude_cal import (
    RoughAmplitudeCal,
    RoughXSXAmplitudeCal,
    EFRoughXSXAmplitudeCal,
    CrossResRoughAmplitudeCal,
)
from .fine_amplitude import FineAmplitudeCal, FineXAmplitudeCal, FineSXAmplitudeCal
from .fine_drag_cal import FineDragCal, FineXDragCal, FineSXDragCal
from .frequency_cal import FrequencyCal
from .fine_frequency_cal import FineFrequencyCal
from .half_angle_cal import HalfAngleCal
