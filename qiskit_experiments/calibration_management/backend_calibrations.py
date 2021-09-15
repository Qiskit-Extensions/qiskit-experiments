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

"""Store and manage the results of calibration experiments in the context of a backend."""

from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import copy

from qiskit.providers.backend import BackendV1 as Backend
from qiskit.circuit import Parameter
from qiskit.pulse import InstructionScheduleMap, ScheduleBlock

from qiskit_experiments.calibration_management.parameter_value import ParameterValue
from qiskit_experiments.calibration_management.calibrations import (
    Calibrations,
    ParameterKey,
    ParameterValueType,
)
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.basis_gate_library import BasisGateLibrary


class FrequencyElement(Enum):
    """An extendable enum for components that have a frequency."""

    QUBIT = "Qubit"
    READOUT = "Readout"


class BackendCalibrations(Calibrations):
    """
    A Calibrations class to enable a seamless interplay with backend objects.
    This class enables users to export their calibrations into a backend object.
    Additionally, it creates frequency parameters for qubits and readout resonators.
    The parameters are named `qubit_lo_freq` and `meas_lo_freq` to be consistent
    with the naming in backend.defaults(). These two parameters are not attached to
    any schedule.
    """

    __qubit_freq_parameter__ = "qubit_lo_freq"
    __readout_freq_parameter__ = "meas_lo_freq"

    def __init__(
        self,
        backend: Backend,
        library: BasisGateLibrary = None,
    ):
        """Setup an instance to manage the calibrations of a backend.

        BackendCalibrations can be initialized from a basis gate library, i.e. a subclass of
        :class:`BasisGateLibrary`. As example consider the following code:

        .. code-block:: python

            cals = BackendCalibrations(
                    backend,
                    library=FixedFrequencyTransmon(
                        basis_gates=["x", "sx"],
                        default_values={duration: 320}
                    )
                )

        Args:
            backend: A backend instance from which to extract the qubit and readout frequencies
                (which will be added as first guesses for the corresponding parameters) as well
                as the coupling map.
            library: A library class that will be instantiated with the library options to then
                get template schedules to register as well as default parameter values.
        """
        if hasattr(backend.configuration(), "control_channels"):
            control_channels = backend.configuration().control_channels
        else:
            control_channels = None

        super().__init__(control_channels)

        # Instruction schedule map variables and support variables.
        self._inst_map = InstructionScheduleMap()
        self._sorted_coupling_map = None

        # Use the same naming convention as in backend.defaults()
        self.qubit_freq = Parameter(self.__qubit_freq_parameter__)
        self.meas_freq = Parameter(self.__readout_freq_parameter__)
        self._register_parameter(self.qubit_freq, ())
        self._register_parameter(self.meas_freq, ())

        if not hasattr(backend.configuration(), "n_qubits"):
            raise CalibrationError("backend.configuration() does not have 'n_qubits'.")

        self._qubits = list(range(backend.configuration().n_qubits))
        self._backend = backend

        for qubit, freq in enumerate(backend.defaults().qubit_freq_est):
            self.add_parameter_value(freq, self.qubit_freq, qubit)

        for meas, freq in enumerate(backend.defaults().meas_freq_est):
            self.add_parameter_value(freq, self.meas_freq, meas)

        if library is not None:

            # Add the basis gates
            for gate in library.basis_gates:
                self.add_schedule(library[gate], n_qubits=library.n_qubits(gate))

            # Add the default values
            for param_conf in library.default_values():
                schedule_name = param_conf[-1]
                if schedule_name in library.basis_gates:
                    self.add_parameter_value(*param_conf)

    @property
    def instruction_schedule_map(self) -> InstructionScheduleMap:
        """Returns the updated instruction schedule map."""
        return self._inst_map

    def _get_frequencies(
        self,
        element: FrequencyElement,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> List[float]:
        """Internal helper method."""

        if element == FrequencyElement.READOUT:
            param = self.meas_freq.name
        elif element == FrequencyElement.QUBIT:
            param = self.qubit_freq.name
        else:
            raise CalibrationError(f"Frequency element {element} is not supported.")

        freqs = []
        for qubit in self._qubits:
            schedule = None  # A qubit frequency is not attached to a schedule.
            if ParameterKey(param, (qubit,), schedule) in self._params:
                freq = self.get_parameter_value(param, (qubit,), schedule, True, group, cutoff_date)
            else:
                if element == FrequencyElement.READOUT:
                    freq = self._backend.defaults().meas_freq_est[qubit]
                elif element == FrequencyElement.QUBIT:
                    freq = self._backend.defaults().qubit_freq_est[qubit]
                else:
                    raise CalibrationError(f"Frequency element {element} is not supported.")

            freqs.append(freq)

        return freqs

    def get_qubit_frequencies(
        self,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> List[float]:
        """
        Get the most recent qubit frequencies. They can be passed to the run-time
        options of :class:`BaseExperiment`. If no calibrated frequency value of a
        qubit is found then the default value from the backend defaults is used.
        Only valid parameter values are returned.

        Args:
            group: The calibration group from which to draw the
                parameters. If not specified, this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values
                that may be erroneous.

        Returns:
            A List of qubit frequencies for all qubits of the backend.
        """
        return self._get_frequencies(FrequencyElement.QUBIT, group, cutoff_date)

    def get_meas_frequencies(
        self,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> List[float]:
        """
        Get the most recent measurement frequencies. They can be passed to the run-time
        options of :class:`BaseExperiment`. If no calibrated frequency value of a
        measurement is found then the default value from the backend defaults is used.
        Only valid parameter values are returned.

        Args:
            group: The calibration group from which to draw the
                parameters. If not specified, this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values
                that may be erroneous.

        Returns:
            A List of measurement frequencies for all qubits of the backend.
        """
        return self._get_frequencies(FrequencyElement.READOUT, group, cutoff_date)

    def export_backend(self) -> Backend:
        """
        Exports the calibrations to a backend object that can be used.

        Returns:
            calibrated backend: A backend with the calibrations in it.
        """
        backend = copy.deepcopy(self._backend)

        backend.defaults().qubit_freq_est = self.get_qubit_frequencies()
        backend.defaults().meas_freq_est = self.get_meas_frequencies()
        backend.default().instruction_schedule_map = self._inst_map

        return backend

    def inst_map_add(
        self,
        instruction_name: str,
        qubits: Tuple[int],
        schedule_name: Optional[str] = None,
        assign_params: Optional[Dict[Union[str, ParameterKey], ParameterValueType]] = None,
    ):
        """Update a single instruction in the instruction schedule map.

        This method can be used to update a single instruction for the given qubits but
        it can also be used by experiments that define custom gates with parameters
        such as the :class:`Rabi` experiment. In a Rabi experiment there is a gate named
        "Rabi" that scans a pulse with a custom amplitude. Therefore we would do

        .. code-block:: python

            cals.inst_map_add("Rabi", (0, ), "xp", assign_params={"amp": Parameter("amp")})

        to temporarily add a pulse for the Rabi gate in the instruction schedule map. This
        then allows calling :code:`transpile(circ, inst_map=cals.instruction_schedule_map)`.

        Args:
            instruction_name: The name of the instruction to add to the instruction schedule map.
            qubits: The qubits to which the instruction will apply.
            schedule_name: The name of the schedule. If None is given then we assume that the
                schedule and the instruction have the same name.
            assign_params: An optional dict of parameter mappings to apply. See for instance
                :meth:`get_schedule` of :class:`Calibrations`.
        """
        schedule_name = schedule_name or instruction_name

        inst_map_args = None
        if assign_params is not None:
            inst_map_args = assign_params.keys()

        self._inst_map.add(
            instruction=instruction_name,
            qubits=qubits,
            schedule=self.get_schedule(schedule_name, qubits, assign_params),
            arguments=inst_map_args,
        )

    def inst_map_remove(self, instruction: str, qubits: Tuple[int]):
        """Remove a single instruction from the inst_map."""
        self._inst_map.remove(instruction, qubits)

    def complete_inst_map_update(self, schedules: Optional[set] = None):
        """Push all schedules from the Calibrations to the inst map.

        This will create instructions with the same name as the schedules.

        Args:
            schedules: The name of the schedules to update. If None is given then
                all schedules will be pushed to instructions.
        """

        for sched_name, _, n_qubits in self._schedules:

            if schedules is not None and sched_name not in schedules:
                continue

            for qubits in self.sorted_coupling_map[n_qubits]:
                try:
                    self._inst_map.add(
                        instruction=sched_name,
                        qubits=qubits,
                        schedule=self.get_schedule(sched_name, qubits),
                    )
                except CalibrationError:
                    # get_schedule may raise an error if not all parameters have values or
                    # default values. In this case we ignore and continue updating inst_map.
                    pass

    def _parameter_inst_map_update(self, param: Parameter):
        """Update all instructions in the inst map that contain the given parameter."""

        schedules = set(key.schedule for key in self._parameter_map_r[param])

        self.complete_inst_map_update(schedules)

    def add_parameter_value(
        self,
        value: Union[int, float, complex, ParameterValue],
        param: Union[Parameter, str],
        qubits: Union[int, Tuple[int, ...]] = None,
        schedule: Union[ScheduleBlock, str] = None,
    ):
        """Wraps :meth:`add_parameter_value` of :class:`Calibrations`.

        This wrapper updates the parameter in the Calibrations but also updates any instructions in
        the instruction schedule map that contain this parameter.

        Args:
            value: The value of the parameter to add. If an int, float, or complex is given
                then the timestamp of the parameter value will automatically be generated
                and set to the current local time of the user.
            param: The parameter or its name for which to add the measured value.
            qubits: The qubits to which this parameter applies.
            schedule: The schedule or its name for which to add the measured parameter value.
        """
        super().add_parameter_value(value, param, qubits, schedule)

        if schedule is not None:
            schedule = schedule.name if isinstance(schedule, ScheduleBlock) else schedule
            param_obj = self.calibration_parameter(param, qubits, schedule)
            self._parameter_inst_map_update(param_obj)

    @property
    def sorted_coupling_map(self) -> Dict[int, List[int]]:
        """Get a sorted coupling map for easy look-up.

        Returns:
            The coupling map in dict format where the key is the number of qubits coupled
            and the value is a list of lists where the sublist shows which qubits are
            coupled. For example a three qubit system with a three qubit gate and three two-
            qubit gates would be represented as

            .. parsed-literal::

                {
                    1: [[0], [1], [2]],
                    2: [[0, 1], [1, 2], [2, 1]],
                    3: [[0, 1, 2]]
                }
        """

        # Use the cached map if there is one.
        if self._sorted_coupling_map is not None:
            return self._sorted_coupling_map

        self._sorted_coupling_map = defaultdict(list)

        # Single qubits
        for qubit in self._qubits:
            self._sorted_coupling_map[1].append([qubit])

        # Multi-qubit couplings
        if self._backend.configuration().coupling_map is not None:
            for coupling in self._backend.configuration().coupling_map:
                self._sorted_coupling_map[len(coupling)].append(coupling)

        return self._sorted_coupling_map
