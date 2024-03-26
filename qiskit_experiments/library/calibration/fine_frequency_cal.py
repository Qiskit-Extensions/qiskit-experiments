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

from typing import Dict, List, Optional, Sequence
import numpy as np

from qiskit.providers.backend import Backend
from qiskit.circuit import QuantumCircuit

from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.calibration_management import BaseCalibrationExperiment
from qiskit_experiments.calibration_management.base_calibrations import BaseCalibrations
from qiskit_experiments.library.characterization.fine_frequency import FineFrequency


class FineFrequencyCal(BaseCalibrationExperiment, FineFrequency):
    """A calibration version of the fine frequency experiment."""

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: BaseCalibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "drive_freq",
        delay_duration: Optional[int] = None,
        repetitions: List[int] = None,
        auto_update: bool = True,
        gate_name: str = "sx",
    ):
        r"""See class :class:`.FineFrequency` for details.

        Note that this class implicitly assumes that the target angle of the gate
        is :math:`\pi/2` as seen from the default analysis options. This experiment
        can be seen as a calibration of a finite duration ``rz(pi/2)`` gate with any
        error attributed to a frequency offset in the qubit.

        Args:
            physical_qubits: Sequence containing the qubit for which to run the
                fine frequency calibration.
            calibrations: The calibrations instance with the schedules.
            backend: Optional, the backend to run the experiment on.
            cal_parameter_name: The name of the parameter to update in the calibrations.
                This defaults to `drive_freq`.
            delay_duration: The duration of the delay at :math:`n=1`. If this value is
                not given then the duration of the gate named ``gate_name`` in the
                calibrations will be used.
            auto_update: Whether to automatically update the calibrations or not. By
                default, this variable is set to True.
            gate_name: This argument is only needed if ``delay_duration`` is None. This
                should be the name of a valid schedule in the calibrations.
        """
        if delay_duration is None:
            delay_duration = calibrations.get_schedule(gate_name, physical_qubits[0]).duration

        super().__init__(
            calibrations,
            physical_qubits,
            delay_duration=delay_duration,
            schedule_name=None,
            repetitions=repetitions,
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )

        if self.backend is not None:
            self.set_experiment_options(dt=self._backend_data.dt)

    @classmethod
    def _default_experiment_options(cls):
        """default values for the fine frequency calibration experiment.

        Experiment Options:
            dt (float): The duration of the time unit ``dt`` of the delay and schedules in seconds.
        """
        options = super()._default_experiment_options()
        options.dt = None
        return options

    def _metadata(self) -> Dict[str, any]:
        """Add metadata to the experiment data making it more self contained.

        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
            delay_duration: The duration of the first delay.
            dt: The number of ``dt`` units of the delay.
        """
        metadata = super()._metadata()
        metadata["delay_duration"] = self.experiment_options.delay_duration
        metadata["dt"] = self.experiment_options.dt
        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            group=self.experiment_options.group,
        )

        return metadata

    def _attach_calibrations(self, circuit: QuantumCircuit):
        """Adds the calibrations to the transpiled circuits."""
        schedule = self._cals.get_schedule("sx", self.physical_qubits)
        circuit.add_calibration("sx", self.physical_qubits, schedule)

    def update_calibrations(self, experiment_data: ExperimentData):
        r"""Update the qubit frequency based on the measured angle deviation.

        The frequency of the qubit is updated according to

        .. math::

            f \to f - \frac{{\rm d}\theta}{2\pi\tau{\rm d}t}

        Here, :math:`{\rm d}\theta` is the measured angle error from the fit. The duration of
        the single qubit-gate is :math:`\tau` in samples and :math:`{\rm d}t` is the duration
        of a sample. This is also the duration of the time unit ``dt`` of the delay.
        """
        result_index = self.experiment_options.result_index
        group = experiment_data.metadata["cal_group"]
        prev_freq = experiment_data.metadata["cal_param_value"]
        tau = experiment_data.metadata["delay_duration"]
        dt = experiment_data.metadata["dt"]

        d_theta = BaseUpdater.get_value(experiment_data, "d_theta", result_index)
        new_freq = prev_freq + d_theta / (2 * np.pi * tau * dt)

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, new_freq, self._param_name, self._sched_name, group
        )
