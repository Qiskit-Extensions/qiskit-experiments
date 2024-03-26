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

"""Rough drag calibration experiment."""

from typing import Dict, Iterable, Optional, Sequence

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.calibration_management import BaseCalibrationExperiment
from qiskit_experiments.calibration_management.base_calibrations import BaseCalibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.library.characterization.drag import RoughDrag


class RoughDragCal(BaseCalibrationExperiment, RoughDrag):
    """A calibration version of the :class:`.RoughDrag` experiment.

    # section: manual
        :ref:`DRAG Calibration`

    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: BaseCalibrations,
        backend: Optional[Backend] = None,
        schedule_name: str = "x",
        betas: Iterable[float] = None,
        cal_parameter_name: Optional[str] = "β",
        auto_update: bool = True,
        group: str = "default",
    ):
        r"""see class :class:`.RoughDrag` for details.

        Args:
            physical_qubits: Sequence containing the qubit for which to run the
                rough DRAG calibration.
            calibrations: The calibrations instance with the schedules.
            backend: Optional, the backend to run the experiment on.
            schedule_name: The name of the schedule to calibrate. Defaults to "x".
            betas: A list of DRAG parameter values to scan. If None is given 51 betas ranging
                from -5 to 5 will be scanned.
            cal_parameter_name: The name of the parameter in the schedule to update.
                Defaults to "β".
            auto_update: Whether or not to automatically update the calibrations. By
                default this variable is set to True.
            group: The group of calibration parameters to use. The default value is "default".
        """
        qubit = physical_qubits[0]
        schedule = calibrations.get_schedule(
            schedule_name, qubit, assign_params={cal_parameter_name: Parameter("β")}, group=group
        )

        self._validate_channels(schedule, [qubit])
        self._validate_parameters(schedule, 1)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule=schedule,
            betas=betas,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )

    def _metadata(self) -> Dict[str, any]:
        """Add metadata to the experiment data making it more self contained.

        The following keys are added to each experiment's metadata:
            cal_param_value: The value of the previous calibrated beta.
            cal_param_name: The name of the parameter in the calibrations.
            cal_schedule: The name of the schedule in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name, self.physical_qubits, self._sched_name, self.experiment_options.group
        )
        return metadata

    def _attach_calibrations(self, circuit: QuantumCircuit):
        """RoughDrag already has the schedules attached in the program circuits."""
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the beta using the value directly reported from the fit.

        See :class:`.DragCalAnalysis` for details on the fit.
        """

        new_beta = BaseUpdater.get_value(
            experiment_data, "beta", self.experiment_options.result_index
        )

        BaseUpdater.add_parameter_value(
            self._cals,
            experiment_data,
            new_beta,
            self._param_name,
            self._sched_name,
            self.experiment_options.group,
        )
