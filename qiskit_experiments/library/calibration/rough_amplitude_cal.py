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

"""Rough amplitude calibration using Rabi."""

from collections import namedtuple
from typing import Dict, Iterable, Optional, Sequence
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.calibration_management import BaseCalibrationExperiment
from qiskit_experiments.calibration_management.base_calibrations import BaseCalibrations
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.library.characterization import Rabi
from qiskit_experiments.calibration_management.update_library import BaseUpdater

AnglesSchedules = namedtuple(
    "AnglesSchedules", ["target_angle", "parameter", "schedule", "previous_value"]
)


class RoughAmplitudeCal(BaseCalibrationExperiment, Rabi):
    """A calibration version of the Rabi experiment."""

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: BaseCalibrations,
        schedule_name: str = "x",
        amplitudes: Iterable[float] = None,
        cal_parameter_name: Optional[str] = "amp",
        target_angle: float = np.pi,
        auto_update: bool = True,
        group: str = "default",
        backend: Optional[Backend] = None,
    ):
        r"""see class :class:`Rabi` for details.

        Args:
            physical_qubits: Sequence containing the qubit for which to run the
                rough amplitude calibration.
            calibrations: The calibrations instance with the schedules.
            schedule_name: The name of the schedule to calibrate. Defaults to "x".
            amplitudes: A list of amplitudes to scan. If None is given 51 amplitudes ranging
                from -0.95 to 0.95 will be scanned.
            cal_parameter_name: The name of the parameter in the schedule to update.
            target_angle: The target angle of the gate to calibrate this will default to a
                :math:`\pi`-pulse.
            auto_update: Whether or not to automatically update the calibrations. By
                default this variable is set to True.
            group: The group of calibration parameters to use. The default value is "default".
            backend: Optional, the backend to run the experiment on.
        """
        qubit = physical_qubits[0]
        schedule = calibrations.get_schedule(
            schedule_name, qubit, assign_params={cal_parameter_name: Parameter("amp")}, group=group
        )

        self._validate_channels(schedule, [qubit])
        self._validate_parameters(schedule, 1)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule=schedule,
            amplitudes=amplitudes,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )

        # Set the pulses to update.
        prev_amp = calibrations.get_parameter_value(cal_parameter_name, qubit, schedule_name)
        self.experiment_options.group = group
        self.experiment_options.angles_schedules = [
            AnglesSchedules(
                target_angle=target_angle,
                parameter=cal_parameter_name,
                schedule=schedule_name,
                previous_value=prev_amp,
            )
        ]

    @classmethod
    def _default_experiment_options(cls):
        """Default values for the rough amplitude calibration experiment.

        Experiment Options:
            result_index (int): The index of the result from which to update the calibrations.
            angles_schedules (list(float, str, str, float)): A list of parameter update information.
                Each entry of the list is a tuple with four entries: the target angle of the
                rotation, the name of the amplitude parameter to update, the name of the schedule
                containing the amplitude parameter to update, and the previous value of the
                amplitude parameter to update. This allows one experiment to update several
                schedules, see for example :class:`RoughXSXAmplitudeCal`.
            group (str): The calibration group to which the parameter belongs. This will default
                to the value "default".
        """
        options = super()._default_experiment_options()

        options.angles_schedules = [
            AnglesSchedules(target_angle=np.pi, parameter="amp", schedule="x", previous_value=None)
        ]

        return options

    def _metadata(self) -> Dict[str, any]:
        """Add metadata to the experiment data making it more self contained.

        The following keys are added to each circuit's metadata:
            angles_schedules: A list of parameter update information. Each entry of the list
                is a tuple with four entries: the target angle of the rotation, the name of the
                amplitude parameter to update, the name of the schedule containing the amplitude
                parameter to update, and the previous value of the amplitude parameter to update.
            cal_group: The calibration group to which the amplitude parameters belong.
        """
        metadata = super()._metadata()
        param_values = []
        for angle, param_name, schedule_name, _ in self.experiment_options.angles_schedules:
            param_val = self._cals.get_parameter_value(
                param_name,
                self._physical_qubits,
                schedule_name,
                group=self.experiment_options.group,
            )

            param_values.append(
                AnglesSchedules(
                    target_angle=angle,
                    parameter=param_name,
                    schedule=schedule_name,
                    previous_value=param_val,
                )
            )

        metadata["angles_schedules"] = param_values

        return metadata

    def _attach_calibrations(self, circuit: QuantumCircuit):
        """Rabi already has the schedules attached in the program circuits."""
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        r"""Update the amplitude of one or several schedules.

        The update rule extracts the rate of the oscillation from the fit to the cosine function.
        Recall that the amplitude is the x-axis in the analysis of the :class:`Rabi` experiment.
        The value of the amplitude is thus the desired rotation-angle divided by the rate of
        the oscillation:

        .. math::

            A_i \to \frac{\theta_i}{\omega}

        where :math:`\theta_i` is the desired rotation angle (e.g. :math:`\pi` and :math:`\pi/2`
        for "x" and "sx" gates, respectively) and :math:`\omega` is the rate of the oscillation.

        Args:
            experiment_data: The experiment data from which to extract the measured Rabi oscillation
            used to set the pulse amplitude.
        """

        result_index = self.experiment_options.result_index
        group = experiment_data.metadata["cal_group"]

        rate = 2 * np.pi * BaseUpdater.get_value(experiment_data, self.__outcome__, result_index)

        for angle, param, schedule, prev_amp in experiment_data.metadata["angles_schedules"]:

            # This implementation conserves the type, while working for both real and complex prev_amp
            value = np.round(angle / rate, decimals=8) * prev_amp / np.abs(prev_amp)

            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, param, schedule, group
            )


class RoughXSXAmplitudeCal(RoughAmplitudeCal):
    """A rough amplitude calibration of x and sx gates."""

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: BaseCalibrations,
        amplitudes: Iterable[float] = None,
        backend: Optional[Backend] = None,
    ):
        """A rough amplitude calibration that updates both the sx and x pulses."""
        super().__init__(
            physical_qubits,
            calibrations,
            schedule_name="x",
            amplitudes=amplitudes,
            backend=backend,
            cal_parameter_name="amp",
            target_angle=np.pi,
        )

        self.experiment_options.angles_schedules = [
            AnglesSchedules(target_angle=np.pi, parameter="amp", schedule="x", previous_value=None),
            AnglesSchedules(
                target_angle=np.pi / 2, parameter="amp", schedule="sx", previous_value=None
            ),
        ]


class EFRoughXSXAmplitudeCal(RoughAmplitudeCal):
    r"""A rough amplitude calibration of :math:`X` and :math:`SX` gates on the
    :math:`|1\rangle` <-> :math:`|2\rangle` transition.
    """

    __outcome__ = "rabi_rate_12"

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: BaseCalibrations,
        amplitudes: Iterable[float] = None,
        backend: Optional[Backend] = None,
        ef_pulse_label: str = "12",
    ):
        r"""A rough amplitude calibration that updates both the sx and x pulses on the
        :math:`|1\rangle` <-> :math:`|2\rangle` transition.

        Args:
            physical_qubits: Sequence containing the index of the qubit
                (technically a qutrit) to run on.
            calibrations: The calibrations instance that stores the pulse schedules.
            amplitudes: The amplitudes to scan.
            backend: Optional, the backend to run the experiment on.
            ef_pulse_label: A label that is post-pended to "x" and "sx" to obtain the name
                of the pulses that drive a :math:`\pi` and :math:`\pi/2` rotation on
                the :math:`|1\rangle` <-> :math:`|2\rangle` transition.
        """
        super().__init__(
            physical_qubits,
            calibrations,
            schedule_name="x" + ef_pulse_label,
            amplitudes=amplitudes,
            backend=backend,
            cal_parameter_name="amp",
            target_angle=np.pi,
        )

        self.experiment_options.angles_schedules = [
            AnglesSchedules(
                target_angle=np.pi,
                parameter="amp",
                schedule="x" + ef_pulse_label,
                previous_value=None,
            ),
            AnglesSchedules(
                target_angle=np.pi / 2,
                parameter="amp",
                schedule="sx" + ef_pulse_label,
                previous_value=None,
            ),
        ]

    def _pre_circuit(self) -> QuantumCircuit:
        """A circuit with operations to perform before the Rabi."""
        circ = QuantumCircuit(1)
        circ.x(0)
        return circ

    def _attach_calibrations(self, circuit: QuantumCircuit):
        """Attach an x calibration if it is defined."""
        # Attach the x calibration as well if it is in self._cals. We allow for
        # it not to be present in case a user wants to rely on the default x
        # calibration and only calibrate the pulses between levels 1 and 2.
        try:
            schedule = self._cals.get_schedule("x", self.physical_qubits)
            circuit.add_calibration("x", self.physical_qubits, schedule)
        except CalibrationError:
            pass
