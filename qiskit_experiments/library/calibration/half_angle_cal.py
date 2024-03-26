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

"""Half angle calibration."""

from typing import Dict, Optional, Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import BaseCalibrationExperiment
from qiskit_experiments.calibration_management.base_calibrations import BaseCalibrations
from qiskit_experiments.library.characterization import HalfAngle
from qiskit_experiments.calibration_management.update_library import BaseUpdater


class HalfAngleCal(BaseCalibrationExperiment, HalfAngle):
    """Calibration version of the :class:`.HalfAngle` experiment."""

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: BaseCalibrations,
        backend: Optional[Backend] = None,
        schedule_name: str = "sx",
        cal_parameter_name: Optional[str] = "angle",
        auto_update: bool = True,
    ):
        """The experiment to update angle of half-pi rotation gates.

        Args:
            physical_qubits: Sequence containing the qubit for which to run the
                half-angle calibration.
            calibrations: The calibrations instance with the schedules.
            backend: Optional, the backend to run the experiment on.
            schedule_name: The name of the schedule to calibrate which defaults to sx.
            cal_parameter_name: The name of the parameter in the schedule to update. This will
                default to 'angle' in accordance with the naming convention of the
                :class:`~qiskit.pulse.ScalableSymbolicPulse` class.
            auto_update:  Whether or not to automatically update the calibrations. By
                default this variable is set to True.

        Raises:
            CalibrationError: if cal_parameter_name is set to ``amp``, to reflect the
                transition from calibrating complex amplitude to calibrating the phase.
            CalibrationError: if the default cal_parameter_name is used, and it is not
                a valid parameter of the calibrated schedule.
        """
        if cal_parameter_name == "amp":
            raise CalibrationError(
                "The Half-Angle calibration experiment was changed from calibrating"
                " the pulse's complex amplitude, to calibrating the angle parameter "
                "in the real (amp,angle) representation. Setting cal_parameter_name to "
                "'amp' thus indicates that you are probably using the experiment in "
                "an inconsistent way. If your pulse does in fact use a complex amplitude,"
                "you need to convert it to (amp,angle) representation, preferably using"
                "the ScalableSymbolicPulse class. Note that all library pulses now use "
                "this representation."
            )
        # If the default cal_parameter_name is used, validate that it is in fact a parameter
        if cal_parameter_name == "angle":
            try:
                calibrations.calibration_parameter("angle", schedule_name=schedule_name)
            except CalibrationError as err:
                raise CalibrationError(
                    "The Half-Angle calibration experiment was changed from calibrating"
                    " the pulse's complex amplitude, to calibrating the angle parameter "
                    "in the real (amp,angle) representation. The default cal_parameter_name "
                    "was thus changed to angle, which is not a valid parameter of the "
                    "calibrated schedule. It is likely that you are trying to calibrate "
                    "a schedule which is defined by a complex amplitude. To use the "
                    "Half-Angle experiment you need to convert the pulses in the schedule "
                    "to (amp,angle) representation (preferably, using the "
                    "ScalableSymbolicPulse class), and have a parameter associated with "
                    "the angle. Note that all library pulses now use this representation."
                ) from err

        super().__init__(
            calibrations,
            physical_qubits,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )

    def _metadata(self) -> Dict[str, any]:
        """Add metadata to the experiment data making it more self-contained.

        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the pulse amplitude. This value together with
                the fit result will be used to find the new value of the pulse amplitude.
            cal_param_name: The name of the parameter in the calibrations.
            cal_schedule: The name of the schedule in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name,
            self._physical_qubits,
            self._sched_name,
            group=self.experiment_options.group,
        )

        return metadata

    def _attach_calibrations(self, circuit: QuantumCircuit):
        """Adds the calibrations to the transpiled circuits."""
        for gate in ["y", "sx"]:
            schedule = self._cals.get_schedule(gate, self.physical_qubits)
            circuit.add_calibration(gate, self.physical_qubits, schedule)

    def update_calibrations(self, experiment_data: ExperimentData):
        r"""Update the value of the parameter in the calibrations.

        The parameter that is updated is the phase of the sx pulse. This phase is contained
        in the complex amplitude of the pulse. The update rule for the half angle calibration is
        therefore:

        .. math::

            A \to A \cdot e^{-i{\rm d}\theta_\text{hac}/2}

        where :math:`A` is the complex amplitude of the sx pulse which has an angle which might be
        different from the angle of the x pulse due to the non-linearity in the mixer's skew. The
        angle :math:`{\rm d}\theta_\text{hac}` is the angle deviation measured through the error
        amplifying pulse sequence.

        Args:
            experiment_data: The experiment data from which to extract the measured over/under
                rotation used to adjust the amplitude.
        """

        result_index = self.experiment_options.result_index
        group = experiment_data.metadata["cal_group"]
        prev_angle = experiment_data.metadata["cal_param_value"]

        d_theta = BaseUpdater.get_value(experiment_data, "d_hac", result_index)
        new_angle = prev_angle - (d_theta / 2)

        BaseUpdater.add_parameter_value(
            self._cals,
            experiment_data,
            new_angle,
            self._param_name,
            self._sched_name,
            group,
        )
