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

"""Fine amplitude calibration experiment."""

from typing import Dict, Optional, Sequence
import numpy as np

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.providers.backend import Backend

from qiskit_experiments.calibration_management import BaseCalibrationExperiment
from qiskit_experiments.calibration_management.base_calibrations import BaseCalibrations
from qiskit_experiments.library.characterization import FineAmplitude
from qiskit_experiments.framework import ExperimentData, Options
from qiskit_experiments.calibration_management.update_library import BaseUpdater


class FineAmplitudeCal(BaseCalibrationExperiment, FineAmplitude):
    r"""A calibration version of the :class:`.FineAmplitude` experiment.

    # section: overview

        :class:`FineAmplitudeCal` is a subclass of :class:`.FineAmplitude`. In the calibration
        experiment the circuits that are run have a custom gate with the pulse schedule attached
        to it through the calibrations.

    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: BaseCalibrations,
        schedule_name: str,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "amp",
        auto_update: bool = True,
        gate: Optional[Gate] = None,
        measurement_qubits: Sequence[int] = None,
    ):
        """See class :class:`FineAmplitude` for details.

        Args:
            physical_qubits: Sequence containing the qubit(s) for which to run
                the fine amplitude calibration. This can be a pair of qubits
                which correspond to control and target qubit.
            calibrations: The calibrations instance with the schedules.
            schedule_name: The name of the schedule to calibrate.
            backend: Optional, the backend to run the experiment on.
            cal_parameter_name: The name of the parameter in the schedule to update.
            auto_update: Whether or not to automatically update the calibrations. By
                default this variable is set to True.
            gate: The gate to repeat in the quantum circuit. If this argument
                is None (the default), then the gate is built from the schedule name.
            measurement_qubits: The qubits in the given physical qubits that need to
                be measured.
        """
        gate = gate or Gate(name=schedule_name, num_qubits=len(physical_qubits), params=[])

        super().__init__(
            calibrations,
            physical_qubits,
            gate,
            schedule_name=schedule_name,
            backend=backend,
            measurement_qubits=measurement_qubits,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )

    @classmethod
    def _default_experiment_options(cls):
        """Default values for the fine amplitude calibration experiment.

        Experiment Options:
            target_angle (float): The target angle of the pulse.
        """
        options = super()._default_experiment_options()
        options.target_angle = np.pi
        return options

    def _metadata(self) -> Dict[str, any]:
        """Add metadata to the experiment data making it more self contained.

        The following keys are added to each experiment's metadata:
            cal_param_value: The value of the pulse amplitude. This value together with
                the fit result will be used to find the new value of the pulse amplitude.
            cal_param_name: The name of the parameter in the calibrations.
            cal_schedule: The name of the schedule in the calibrations.
            target_angle: The target angle of the gate.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["target_angle"] = self.experiment_options.target_angle
        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            self._sched_name,
            group=self.experiment_options.group,
        )

        return metadata

    def _attach_calibrations(self, circuit: QuantumCircuit):
        """Adds the calibrations to the transpiled circuits."""
        for gate in ["x", "sx"]:
            schedule = self._cals.get_schedule(gate, self.physical_qubits)
            circuit.add_calibration(gate, self.physical_qubits, schedule)

    def update_calibrations(self, experiment_data: ExperimentData):
        r"""Update the amplitude of the pulse in the calibrations.

        The update rule of this experiment is

        .. math::

            A \to A \frac{\theta_\text{target}}{\theta_\text{target} + {\rm d}\theta}

        Where :math:`A` is the amplitude of the pulse before the update.

        Args:
            experiment_data: The experiment data from which to extract the measured over/under
                rotation used to adjust the amplitude.
        """

        result_index = self.experiment_options.result_index
        group = experiment_data.metadata["cal_group"]
        target_angle = experiment_data.metadata["target_angle"]
        prev_amp = experiment_data.metadata["cal_param_value"]

        # Protect against cases where the complex amplitude was converted to a list.
        if isinstance(prev_amp, list) and len(prev_amp) == 2:
            prev_amp = prev_amp[0] + 1.0j * prev_amp[1]

        d_theta = BaseUpdater.get_value(experiment_data, "d_theta", result_index)

        BaseUpdater.add_parameter_value(
            self._cals,
            experiment_data,
            prev_amp * target_angle / (target_angle + d_theta),
            self._param_name,
            self._sched_name,
            group,
        )


class FineXAmplitudeCal(FineAmplitudeCal):
    """A calibration experiment to calibrate the amplitude of the X schedule."""

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: BaseCalibrations,
        schedule_name: str,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "amp",
        auto_update: bool = True,
    ):
        super().__init__(
            physical_qubits,
            calibrations,
            schedule_name,
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi,
                "phase_offset": np.pi / 2,
            }
        )

    @classmethod
    def _default_transpile_options(cls):
        """Default transpile options.

        Transpile Options:
            basis_gates (list(str)): A list of basis gates needed for this experiment.
                The schedules for these basis gates will be provided by the instruction
                schedule map from the calibrations.
        """
        options = super()._default_transpile_options()
        options.basis_gates = ["x", "sx"]

        return options

    def _pre_circuit(self, num_clbits: int) -> QuantumCircuit:
        """The preparation circuit is an sx gate to move to the equator of the Bloch sphere."""
        circuit = QuantumCircuit(self.num_qubits, num_clbits)
        circuit.sx(0)
        return circuit


class FineSXAmplitudeCal(FineAmplitudeCal):
    """A calibration experiment to calibrate the amplitude of the SX schedule."""

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: BaseCalibrations,
        schedule_name: str,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "amp",
        auto_update: bool = True,
    ):
        super().__init__(
            physical_qubits,
            calibrations,
            schedule_name,
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi / 2,
                "phase_offset": np.pi,
            }
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            add_sx (bool): This option is False by default when calibrating gates with a target
                angle per gate of :math:`\pi/2` as this increases the sensitivity of the
                experiment.
            add_xp_circuit (bool): This option is False by default when calibrating gates with
                a target angle per gate of :math:`\pi/2`.
            repetitions (List[int]): By default the repetitions take on odd numbers for
                :math:`\pi/2` target angles as this ideally prepares states on the equator of
                the Bloch sphere. Note that the repetitions include two repetitions which
                plays the same role as including a circuit with an X gate.
            target_angle (float): The target angle per gate.
        """
        options = super()._default_experiment_options()
        options.add_cal_circuits = False
        options.repetitions = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]
        options.target_angle = np.pi / 2
        return options
