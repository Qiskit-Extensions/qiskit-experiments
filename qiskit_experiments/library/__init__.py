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
======================================================
Experiment Library (:mod:`qiskit_experiments.library`)
======================================================

.. currentmodule:: qiskit_experiments.library

A library of of quantum characterization, calibration and verification
experiments for calibrating and benchmarking quantum devices. See
:mod:`qiskit_experiments.framework` for general information on the framework
for running experiments.


.. _verification:

Verification Experiments
========================

Experiments for verification and validation of quantum devices.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ~randomized_benchmarking.StandardRB
    ~randomized_benchmarking.InterleavedRB
    ~tomography.TomographyExperiment
    ~tomography.StateTomography
    ~tomography.ProcessTomography
    ~tomography.MitigatedStateTomography
    ~tomography.MitigatedProcessTomography
    ~quantum_volume.QuantumVolume

.. _characterization single qubit:

Characterization Experiments: Single Qubit
==========================================

Experiments for characterization of properties of individual qubits.
Some experiments also have a calibration experiment version.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ~characterization.T1
    ~characterization.T2Hahn
    ~characterization.T2Ramsey
    ~characterization.Tphi
    ~characterization.QubitSpectroscopy
    ~characterization.EFSpectroscopy
    ~characterization.HalfAngle
    ~characterization.FineAmplitude
    ~characterization.FineXAmplitude
    ~characterization.FineSXAmplitude
    ~characterization.Rabi
    ~characterization.EFRabi
    ~characterization.CrossResRabi
    ~characterization.RamseyXY
    ~characterization.FineFrequency
    ~characterization.ReadoutAngle
    ~characterization.ResonatorSpectroscopy
    ~characterization.RoughDrag
    ~characterization.FineDrag
    ~characterization.FineXDrag
    ~characterization.FineSXDrag
    ~characterization.MultiStateDiscrimination
    ~characterization.StarkRamseyXY

.. _characterization two qubits:

Characterization Experiments: Two Qubits
========================================

Experiments for characterization of properties of two qubit interactions.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ~characterization.CrossResonanceHamiltonian
    ~characterization.EchoedCrossResonanceHamiltonian
    ~characterization.ZZRamsey
    ~characterization.FineZXAmplitude

.. _characterization-mitigation:

Characterization Experiments: Mitigation
========================================

Experiments for characterizing and mitigating readout error.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ~characterization.LocalReadoutError
    ~characterization.CorrelatedReadoutError

.. _calibration:

Calibration Experiments
=======================

Experiments for pulse level calibration of quantum gates. These experiments
are usually run with a
:class:`~qiskit_experiments.calibration_management.Calibrations`
class instance to manage parameters and pulse schedules.
See :doc:`/tutorials/calibrations` for examples.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ~calibration.RoughFrequencyCal
    ~calibration.RoughEFFrequencyCal
    ~calibration.FrequencyCal
    ~calibration.FineFrequencyCal
    ~calibration.RoughDragCal
    ~calibration.FineXDragCal
    ~calibration.FineSXDragCal
    ~calibration.FineDragCal
    ~calibration.FineAmplitudeCal
    ~calibration.FineXAmplitudeCal
    ~calibration.FineSXAmplitudeCal
    ~calibration.HalfAngleCal
    ~calibration.RoughAmplitudeCal
    ~calibration.RoughXSXAmplitudeCal
    ~calibration.EFRoughXSXAmplitudeCal
    ~calibration.CrossResRoughAmplitudeCal

"""
from .calibration import (
    RoughDragCal,
    FineDragCal,
    FineXDragCal,
    FineSXDragCal,
    RoughAmplitudeCal,
    RoughXSXAmplitudeCal,
    EFRoughXSXAmplitudeCal,
    CrossResRoughAmplitudeCal,
    FineAmplitudeCal,
    FineXAmplitudeCal,
    FineSXAmplitudeCal,
    RoughFrequencyCal,
    RoughEFFrequencyCal,
    FrequencyCal,
    FineFrequencyCal,
    HalfAngleCal,
)
from .characterization import (
    T1,
    T2Hahn,
    T2Ramsey,
    Tphi,
    QubitSpectroscopy,
    EFSpectroscopy,
    CrossResonanceHamiltonian,
    EchoedCrossResonanceHamiltonian,
    RoughDrag,
    FineDrag,
    FineXDrag,
    FineSXDrag,
    Rabi,
    EFRabi,
    CrossResRabi,
    HalfAngle,
    FineAmplitude,
    FineXAmplitude,
    FineSXAmplitude,
    FineZXAmplitude,
    RamseyXY,
    FineFrequency,
    ReadoutAngle,
    ResonatorSpectroscopy,
    LocalReadoutError,
    CorrelatedReadoutError,
    ZZRamsey,
    MultiStateDiscrimination,
    StarkRamseyXY,
)
from .randomized_benchmarking import StandardRB, InterleavedRB
from .tomography import (
    TomographyExperiment,
    StateTomography,
    ProcessTomography,
    MitigatedStateTomography,
    MitigatedProcessTomography,
)
from .quantum_volume import QuantumVolume

# Experiment Sub-modules
from . import calibration
from . import characterization
from . import randomized_benchmarking
from . import tomography
from . import quantum_volume
