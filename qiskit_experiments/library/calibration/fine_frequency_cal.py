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

"""Fine frequency calibration experiment."""

from typing import List, Optional
import numpy as np

from qiskit.providers.backend import Backend

from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.calibration_management import (
    BaseCalibrationExperiment,
    BackendCalibrations,
)
from qiskit_experiments.library.characterization.fine_frequency import FineFrequency


class FineFrequencyCal(BaseCalibrationExperiment, FineFrequency):
    """A calibration version of the fine frequency experiment."""

    def __init__(
        self,
        qubit: int,
        calibrations: BackendCalibrations,
        backend: Optional[Backend] = None,
        repetitions: List[int] = None,
        auto_update: bool = True,
    ):
        r"""see class :class:`FineDrag` for details.

        Note that this class implicitly assumes that the target angle of the gate
        is :math:`\pi` as seen from the default experiment options.

        Args:
            qubit: The qubit for which to run the fine frequency calibration.
            calibrations: The calibrations instance with the schedules.
            backend: Optional, the backend to run the experiment on.
            auto_update: Whether or not to automatically update the calibrations. By
                default this variable is set to True.
        """
        super().__init__(
            calibrations,
            qubit,
            repetitions,
            schedule_name=None,
            backend=backend,
            cal_parameter_name="qubit_lo_freq",
            auto_update=auto_update,
        )

        self.set_transpile_options(
            inst_map=calibrations.default_inst_map,
            basis_gates=["sx", "rz"],
        )

        if self.backend is not None:
            self.set_experiment_options(dt=getattr(self.backend.configuration(), "dt", None))

    @classmethod
    def _default_experiment_options(cls):
        """default values for the fine frequency calibration experiment.

        Experiment Options:
            dt (float): The duration of the samples of the arbitrary waveform generator that
                is used to generate the pulses.
        """
        options = super()._default_experiment_options()
        options.dt = None
        return options

    def _add_cal_metadata(self, experiment_data: ExperimentData):
        """Add metadata to the experiment data making it more self contained.

        The following keys are added to each circuit's metadata:
            cal_param_value: The value of the drag parameter. This value together with
                the fit result will be used to find the new value of the drag parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_schedule: The name of the schedule in the calibrations.
            target_angle: The target angle of the gate.
            cal_group: The calibration group to which the parameter belongs.
        """

        param_val = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            group=self.experiment_options.group,
        )

        experiment_data.metadata.update(
            {
                "cal_param_value": param_val,
                "cal_param_name": self._param_name,
                "cal_group": self.experiment_options.group,
                "single_qubit_gate_duration": self.experiment_options.sq_gate_duration,
                "dt": self.experiment_options.dt,
            }
        )

    def update_calibrations(self, experiment_data: ExperimentData):
        r"""Update the qubit frequency based on the measured angle deviation.

        The frequency of the qubit is updated according to

        ..math::

            f \to f - \frac{{\rm d}\theta}{2\pi\tau{\rm d}t}

        Here, :math:`{\rm d}\theta` is the measured angle error from the fit. The duration of
        the single qubit-gate is :math:`\tau` in samples and :math:`{\rm d}t` is the duration
        of a single arbitrary waveform generator sample.
        """
        result_index = self.experiment_options.result_index
        group = experiment_data.metadata["cal_group"]
        prev_freq = experiment_data.metadata["cal_param_value"]
        tau = experiment_data.metadata["single_qubit_gate_duration"]
        dt = experiment_data.metadata["dt"]

        d_theta = BaseUpdater.get_value(experiment_data, "d_theta", result_index)
        new_freq = prev_freq - d_theta / (2 * np.pi * tau * dt)

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, new_freq, self._param_name, self._sched_name, group
        )
