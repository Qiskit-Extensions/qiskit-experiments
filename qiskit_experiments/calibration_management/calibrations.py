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

"""Class to store and manage the results of calibration experiments."""

import warnings
import os
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import Any, Dict, Set, Tuple, Union, List, Optional
import csv
import dataclasses
import json
import rustworkx as rx

from qiskit.pulse import (
    ScheduleBlock,
    DriveChannel,
    ControlChannel,
    MeasureChannel,
    AcquireChannel,
    RegisterSlot,
    MemorySlot,
    InstructionScheduleMap,
)
from qiskit.pulse.channels import PulseChannel
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.providers.backend import Backend
from qiskit.utils.deprecation import deprecate_func, deprecate_arg

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.base_calibrations import BaseCalibrations
from qiskit_experiments.calibration_management.basis_gate_library import BasisGateLibrary
from qiskit_experiments.calibration_management.parameter_value import ParameterValue
from qiskit_experiments.calibration_management.control_channel_map import ControlChannelMap
from qiskit_experiments.calibration_management.calibration_utils import (
    used_in_references,
    validate_channels,
    reference_info,
    update_schedule_dependency,
)
from qiskit_experiments.calibration_management.calibration_key_types import (
    ParameterKey,
    ParameterValueType,
    ScheduleKey,
)
from qiskit_experiments.framework import BackendData, ExperimentEncoder, ExperimentDecoder


class Calibrations(BaseCalibrations):
    """
    A class to manage schedules with calibrated parameter values. Schedules are
    intended to be fully parameterized, including the index of the channels. See
    the module-level documentation for extra details. Note that only instances of
    ScheduleBlock are supported.
    """

    def __init__(
        self,
        coupling_map: Optional[List[List[int]]] = None,
        control_channel_map: Optional[Dict[Tuple[int, ...], List[ControlChannel]]] = None,
        libraries: Optional[List[BasisGateLibrary]] = None,
        add_parameter_defaults: bool = True,
        backend_name: Optional[str] = None,
        backend_version: Optional[str] = None,
    ):
        """Initialize the calibrations.

        Calibrations can be initialized from a list of basis gate libraries, i.e. a subclass of
        :class:`.BasisGateLibrary`. As example consider the following code:

        .. code-block:: python

            cals = Calibrations(
                    libraries=[
                        FixedFrequencyTransmon(
                            basis_gates=["x", "sx"],
                            default_values={duration: 320}
                        )
                    ]
                )

        Args:
            coupling_map: The device's coupling map. Each sub-list describes connected qubits
                For example, the coupling map of a fully pairwise-connected backend with three
                qubits is :code:`[[0, 1], [1, 0], [1, 2], [2, 1], [2, 0], [0, 2]]`.
            control_channel_map: A configuration dictionary of any control channels. The
                keys are tuples of qubits and the values are a list of ControlChannels
                that correspond to the qubits in the keys. If a control_channel_map is given
                then the qubits must be in the coupling_map.
            libraries: A list of library instances from which to get template schedules to register
                as well as default parameter values.
            add_parameter_defaults: A boolean to indicate weather the default parameter values of
                the given libraries should be used to populate the calibrations. By default this
                value is True but can be set to false when deserializing a calibrations object.
            backend_name: The name of the backend that these calibrations are attached to.
            backend_version: The version of the backend that these calibrations are attached to.
        """
        self._backend_name = backend_name
        self._backend_version = backend_version

        # Mapping between qubits and their control channels.
        self._control_channel_map = control_channel_map if control_channel_map else {}

        # Store the reverse mapping between control channels and qubits for ease of look-up.
        self._controls_config_r = {}
        for qubits, channels in self._control_channel_map.items():
            for channel in channels:
                self._controls_config_r[channel] = qubits

        # Dict of the form: (schedule.name, parameter.name, qubits): Parameter
        self._parameter_map = {}

        # Reverse mapping of _parameter_map
        self._parameter_map_r = defaultdict(set)

        # Default dict of the form: (schedule.name, parameter.name, qubits): [ParameterValue, ...]
        self._params = defaultdict(list)

        # Dict of the form: ScheduleKey: ScheduleBlock
        self._schedules = {}

        # Dict of the form: ScheduleKey: int (number of qubits in corresponding circuit instruction)
        self._schedules_qubits = {}

        # A directed acyclic graph to manage schedule reference dependencies.
        # Each node is a schedule key and edges are schedule references.
        # The acyclic nature prevents users from registering schedules with cyclic dependencies.
        self._schedule_dependency = rx.PyDiGraph(check_cycle=True)

        # A variable to store all parameter hashes encountered and present them as ordered
        # indices to the user.
        self._hash_to_counter_map = {}
        self._parameter_counter = 0

        self._libraries = libraries
        if libraries is not None:
            if not isinstance(libraries, list):
                libraries = [libraries]

            for lib in libraries:
                # Add the basis gates
                for gate in lib.basis_gates:
                    self.add_schedule(lib[gate], num_qubits=lib.num_qubits(gate))

                # Add the default values
                if add_parameter_defaults:
                    for param_conf in lib.default_values():
                        self.add_parameter_value(*param_conf, update_inst_map=False)

                # Add the parameters that do not belong to a schedule.
                for param_name in lib.__parameters_without_schedule__:
                    self._register_parameter(Parameter(param_name), tuple())

        # This internal parameter is False so that if a schedule is added after the
        # init it will be set to True and serialization will raise an error.
        self._has_manually_added_schedule = False

        # Instruction schedule map variables and support variables.
        self._inst_map = InstructionScheduleMap()

        # Backends with a single qubit may not have a coupling map.
        self._coupling_map = coupling_map if coupling_map is not None else []

        # A dict extension of the coupling map where the key is the number of qubits and
        # the values are a list of qubits coupled.
        self._operated_qubits = self._get_operated_qubits()
        self._check_consistency()

        # Push the schedules to the instruction schedule map.
        self.update_inst_map()

    @property
    @deprecate_func(
        is_property=True,
        since="0.6",
        package_name="qiskit-experiments",
        additional_msg="The drive_freq is moved to FixedFrequencyTransmon basis gate library.",
    )
    def drive_freq(self):
        """Parameter object for qubit drive frequency."""
        return self._parameter_map.get(("drive_freq", (), None), None)

    @property
    @deprecate_func(
        is_property=True,
        since="0.6",
        package_name="qiskit-experiments",
        additional_msg="The meas_freq is moved to FixedFrequencyTransmon basis gate library.",
    )
    def meas_freq(self):
        """Parameter object for qubit measure frequency."""
        return self._parameter_map.get(("meas_freq", (), None), None)

    def _check_consistency(self):
        """Check that the attributes defined in self are consistent.

        Raises:
            CalibrationError: If there is a control channel map but no coupling map.
            CalibrationError: If a qubit in the control channel map is not in the
                coupling map.
        """
        if not self._coupling_map and self._control_channel_map:
            raise CalibrationError("No coupling map but a control channel map was found.")

        if self._coupling_map and self._control_channel_map:
            cmap_qubits = set(qubit for pair in self._coupling_map for qubit in pair)
            for qubits in self._control_channel_map:
                if not set(qubits).issubset(cmap_qubits):
                    raise CalibrationError(
                        f"Qubits {qubits} of control_channel_map are not in the coupling map."
                    )

    @property
    def backend_name(self) -> str:
        """Return the name of the backend."""
        return self._backend_name

    @property
    def backend_version(self) -> str:
        """Return the version of the backend."""
        return self._backend_version

    @property
    def schedule_dependency(self) -> rx.PyDiGraph:
        """Return the schedule dependencies in the calibrations."""
        return self._schedule_dependency.copy()

    @classmethod
    def from_backend(
        cls,
        backend: Backend,
        libraries: Optional[List[BasisGateLibrary]] = None,
        add_parameter_defaults: bool = True,
    ) -> "Calibrations":
        """Create an instance of Calibrations from a backend.

        Args:
            backend: A backend instance from which to extract the qubit and readout frequencies
                (which will be added as first guesses for the corresponding parameters) as well
                as the coupling map.
            libraries: A list of libraries from which to get template schedules to register as
                well as default parameter values.
            add_parameter_defaults: A boolean to indicate whether the default parameter values of
                the given library should be used to populate the calibrations. By default, this
                value is ``True``.

        Returns:
            An instance of Calibrations instantiated from a backend.
        """
        backend_data = BackendData(backend)

        control_channel_map = {}
        if backend_data.coupling_map is not None:
            for qargs in backend_data.coupling_map:
                control_channel_map[tuple(qargs)] = backend_data.control_channel(qargs)

        cals = Calibrations(
            backend_data.coupling_map,
            control_channel_map,
            libraries,
            add_parameter_defaults,
            backend_data.name,
            backend_data.version,
        )

        if add_parameter_defaults:
            if ("drive_freq", (), None) in cals._parameter_map:
                for qubit, freq in enumerate(backend_data.drive_freqs):
                    cals.add_parameter_value(freq, "drive_freq", qubit, update_inst_map=False)

            if ("meas_freq", (), None) in cals._parameter_map:
                for meas, freq in enumerate(backend_data.meas_freqs):
                    cals.add_parameter_value(freq, "meas_freq", meas, update_inst_map=False)

        # Update the instruction schedule map after adding all parameter values.
        cals.update_inst_map()

        return cals

    @property
    def libraries(self) -> Optional[List[BasisGateLibrary]]:
        """Return the libraries used to initialize the calibrations."""
        return self._libraries

    def _get_operated_qubits(self) -> Dict[int, List[int]]:
        """Get a dict describing qubit couplings.

        This is an extension of the coupling map and used as a convenience to help populate
        the instruction schedule map.

        Returns:
            A dict where the key is the number of qubits coupled and the value is a list of
            lists where the sublist shows which qubits are coupled. For example, a three qubit
            system with a three qubit gate and three two-qubit gates would be represented as

            .. parsed-literal::

                {
                    1: [[0], [1], [2]],
                    2: [[0, 1], [1, 2], [2, 1]],
                    3: [[0, 1, 2]]
                }
        """
        operated_qubits = defaultdict(list)

        # Single qubits
        if self._coupling_map:
            for qubit in set(qubit for coupled in self._coupling_map for qubit in coupled):
                operated_qubits[1].append([qubit])
        else:
            # Edge case for single-qubit device.
            operated_qubits[1].append([0])

        # Multi-qubit couplings
        for coupling in self._coupling_map:
            operated_qubits[len(coupling)].append(coupling)

        return operated_qubits

    @property
    def default_inst_map(self) -> InstructionScheduleMap:
        """Return the default and up to date instruction schedule map."""
        return self._inst_map

    def get_inst_map(
        self,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> InstructionScheduleMap:
        """Get an Instruction schedule map with the calibrated pulses.

        If the group is 'default' and cutoff date is None then the automatically updated
        instruction schedule map is returned. However, if these values are different then
        a new instruction schedule map is populated based on the values.

        Args:
            group: The calibration group from which to draw the parameters.
                If not specified this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.

        Returns:
            An instruction schedule map with parameters updated up to the desired cutoff date
            and from the desired calibration group.
        """
        if group == "default" and cutoff_date is None:
            return self._inst_map

        inst_map = InstructionScheduleMap()

        self.update_inst_map(group=group, cutoff_date=cutoff_date, inst_map=inst_map)

        return inst_map

    def update_inst_map(
        self,
        schedules: Optional[Set[str]] = None,
        qubits: Optional[Tuple[int, ...]] = None,
        group: Optional[str] = "default",
        cutoff_date: datetime = None,
        inst_map: Optional[InstructionScheduleMap] = None,
    ):
        """Push all schedules from the Calibrations to the inst map.

        This will create instructions with the same name as the schedules.

        Args:
            schedules: The name of the schedules to update. If None is given then
                all schedules will be pushed to instructions.
            qubits: The qubits for which to update the instruction schedule map.
                If qubits is None then all possible schedules defined by the coupling
                map will be updated. Note that this argument specifies a particular set of
                qubits to update instructions for. For example, if qubits is :code:`(2, 3)` then
                only two-qubit instructions that apply to qubits 2 and 3 will be updated. Here,
                single-qubit instructions will not be updated.
            group: The calibration group from which to draw the parameters. If not specified
                this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.
            inst_map: The instruction schedule map to update. If None is given then the default
                instruction schedule map (i.e. self._inst_map) will be updated.
        """
        inst_map = inst_map or self._inst_map

        for key in self._schedules:
            sched_name = key.schedule

            if schedules is not None and sched_name not in schedules:
                continue

            if qubits:
                self._robust_inst_map_add(inst_map, sched_name, qubits, group, cutoff_date)
            else:
                for qubits_ in self._operated_qubits[self._schedules_qubits[key]]:
                    self._robust_inst_map_add(inst_map, sched_name, qubits_, group, cutoff_date)

    def _robust_inst_map_add(
        self,
        inst_map: InstructionScheduleMap,
        sched_name: str,
        qubits: Union[int, Tuple[int, ...]],
        group: str,
        cutoff: datetime,
    ):
        """A helper method for update_inst_map.

        get_schedule may raise an error if not all parameters have values or
        default values. In this case we ignore and continue updating inst_map.
        Note that ``qubits`` may only be a sub-set of the qubits of the schedule that
        we want to update. This may arise in cases such as an ECR gate schedule that calls
        an X-gate schedule. When updating the X-gate schedule we need to also update the
        corresponding ECR schedules which operate on a larger number of qubits.

        Args:
            sched_name: The name of the schedule.
            qubits: The qubit to which the schedule applies. Note, these may be only a
                subset of the qubits in the schedule. For example, if the name of the
                schedule is `"cr"` we may have `qubits` be `(3, )` and this function
                will update the CR schedules on all schedules which involve qubit 3.
            group: The calibration group.
            cutoff: The cutoff date.
        """
        for update_qubits in self._get_full_qubits_of_schedule(sched_name, qubits):
            try:
                schedule = self.get_schedule(
                    sched_name, update_qubits, group=group, cutoff_date=cutoff
                )
                inst_map.add(instruction=sched_name, qubits=update_qubits, schedule=schedule)
            except CalibrationError:
                # get_schedule may raise an error if not all parameters have values or
                # default values. In this case we ignore and continue updating inst_map.
                pass

    def _get_full_qubits_of_schedule(
        self, schedule_name: str, partial_qubits: Tuple[int, ...]
    ) -> List[Tuple[int, ...]]:
        """Find all qubits for which there is a schedule ``schedule_name`` on ``partial_qubits``.

        This method uses the map between the schedules and the number of qubits that they
        operate on as well as the extension of the coupling map ``_operated_qubits`` to find
        which qubits are involved in the schedule named ``schedule_name`` involving the
        ``partial_qubits``.

        Args:
            schedule_name: The name of the schedule as registered in ``self``.
            partial_qubits: A sub-set of qubits on which the schedule applies.

        Returns:
            A list of tuples. Each tuple is the set of qubits for which there is a schedule
            named ``schedule_name`` and ``partial_qubits`` is a sub-set of said qubits.
        """
        for key, circuit_inst_num_qubits in self._schedules_qubits.items():
            if key.schedule == schedule_name:
                if len(partial_qubits) == circuit_inst_num_qubits:
                    return [partial_qubits]

                else:
                    candidates = self._operated_qubits[circuit_inst_num_qubits]
                    qubits_for_update = []
                    for candidate_qubits in candidates:
                        if set(partial_qubits).issubset(set(candidate_qubits)):
                            qubits_for_update.append(tuple(candidate_qubits))

                    return qubits_for_update

        return []

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
        then allows calling :code:`transpile(circ, inst_map=cals.default_inst_map)`.

        Args:
            instruction_name: The name of the instruction to add to the instruction schedule map.
            qubits: The qubits to which the instruction will apply.
            schedule_name: The name of the schedule. If None is given then we assume that the
                schedule and the instruction have the same name.
            assign_params: An optional dict of parameter mappings to apply. See for instance
                :meth:`.get_schedule` of :class:`.Calibrations`.
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

    def add_schedule(
        self,
        schedule: ScheduleBlock,
        qubits: Union[int, Tuple[int, ...]] = None,
        num_qubits: Optional[int] = None,
    ):
        """Add a schedule block and register its parameters.

        Schedules that use Call instructions must register the called schedules separately.

        Args:
            schedule: The :class:`ScheduleBlock` to add.
            qubits: The qubits for which to add the schedules. If None or an empty tuple is
                given then this schedule is the default schedule for all qubits and, in this
                case, the number of qubits that this schedule act on must be given.
            num_qubits: The number of qubits that this schedule will act on when exported to
                a circuit instruction. This argument is optional as long as qubits is either
                not None or not an empty tuple (i.e. default schedule).

        Raises:
            CalibrationError: If schedule is not an instance of :class:`ScheduleBlock`.
            CalibrationError: If a schedule has assigned references.
            CalibrationError: If several parameters in the same schedule have the same name.
            CalibrationError: If the schedule name starts with the prefix of ScheduleBlock.
            CalibrationError: If the schedule calls subroutines that have not been registered.
            CalibrationError: If a :class:`Schedule` is Called instead of a :class:`ScheduleBlock`.
            CalibrationError: If a schedule with the same name exists and acts on a different
                number of qubits.

        """
        self._has_manually_added_schedule = True

        qubits = self._to_tuple(qubits)

        if len(qubits) == 0 and num_qubits is None:
            raise CalibrationError("Both qubits and num_qubits cannot simultaneously be None.")

        num_qubits = len(qubits) or num_qubits

        if not isinstance(schedule, ScheduleBlock):
            raise CalibrationError(f"{schedule.name} is not a ScheduleBlock.")

        if len(schedule.references) != len(schedule.references.unassigned()):
            raise CalibrationError(
                f"Cannot add {schedule} with assigned references. {self.__class__.__name__} only "
                "accepts schedules without references or schedules with references by name."
            )

        sched_key = ScheduleKey(schedule.name, qubits)

        # Ensure one to one mapping between name and number of qubits.
        if sched_key in self._schedules_qubits and self._schedules_qubits[sched_key] != num_qubits:
            raise CalibrationError(
                f"Cannot add schedule {schedule.name} acting on {num_qubits} qubits."
                "self already contains a schedule with the same name acting on "
                f"{self._schedules_qubits[sched_key]} qubits. Remove old schedule first."
            )

        # check that channels, if parameterized, have the proper name format.
        if schedule.name.startswith(ScheduleBlock.prefix):
            raise CalibrationError(
                f"{self.__class__.__name__} uses the `name` property of the schedule as part of a "
                f"database key. Using the automatically generated name {schedule.name} may have "
                f"unintended consequences. Please define a meaningful and unique schedule name."
            )

        param_indices = validate_channels(schedule)

        # Check that subroutines are present.
        for reference in schedule.references:
            self.get_template(*reference_info(reference, qubits))

        # Clean the parameter to schedule mapping. This is needed if we overwrite a schedule.
        self._clean_parameter_map(schedule.name, qubits)

        # Add the schedule.
        self._schedules[sched_key] = schedule
        self._schedules_qubits[sched_key] = num_qubits

        # Update the schedule dependency.
        update_schedule_dependency(schedule, self._schedule_dependency, sched_key)

        # Register parameters that are not indices.
        params_to_register = set()
        for param in schedule.parameters:
            if param not in param_indices:
                params_to_register.add(param)

        if len(params_to_register) != len(set(param.name for param in params_to_register)):
            raise CalibrationError(f"Parameter names in {schedule.name} must be unique.")

        for param in params_to_register:
            self._register_parameter(param, qubits, schedule)

    def has_template(self, schedule_name: str, qubits: Optional[Tuple[int, ...]] = None) -> bool:
        """Test if a template schedule is defined

        Args:
            schedule_name: The name of the template schedule.
            qubits: The qubits under which the template schedule was registered.

        Returns:
            True if a template exists for the schedule name for the given qubits
        """
        found = False
        try:
            self.get_template(schedule_name, qubits)
            found = True
        except CalibrationError:
            pass

        return found

    def get_template(
        self, schedule_name: str, qubits: Optional[Tuple[int, ...]] = None
    ) -> ScheduleBlock:
        """Get a template schedule.

        Allows the user to get a template schedule that was previously registered.
        A template schedule will typically be fully parametric, i.e. all pulse
        parameters and channel indices are represented by :class:`Parameter`.

        Args:
            schedule_name: The name of the template schedule.
            qubits: The qubits under which the template schedule was registered.

        Returns:
            The registered template schedule.

        Raises:
            CalibrationError: If no template schedule for the given schedule name and qubits
                was registered.
        """
        key = ScheduleKey(schedule_name, self._to_tuple(qubits))

        if key in self._schedules:
            return self._schedules[key]

        if ScheduleKey(schedule_name, ()) in self._schedules:
            return self._schedules[ScheduleKey(schedule_name, ())]

        if qubits:
            msg = f"Could not find schedule {schedule_name} on qubits {qubits}."
        else:
            msg = f"Could not find schedule {schedule_name}."

        raise CalibrationError(msg)

    def remove_schedule(self, schedule: ScheduleBlock, qubits: Union[int, Tuple[int, ...]] = None):
        """Remove a schedule that was previously registered.

        Allows users to remove a schedule from the calibrations. The history of the parameters
        will remain in the calibrations.

        Args:
            schedule: The schedule to remove.
            qubits: The qubits for which to remove the schedules. If None is given then this
                schedule is the default schedule for all qubits.

        Raises:
            CalibrationError: If other schedules depend on ``schedule``.
        """
        qubits = self._to_tuple(qubits)
        sched_key = ScheduleKey(schedule.name, qubits)

        # Remove the schedule from the schedule dependency DAG. Raise if others depend on it.
        sched_idx = self._schedule_dependency.nodes().index(sched_key)
        prev_nodes = self._schedule_dependency.predecessors(sched_idx)
        if len(prev_nodes) > 0:
            raise CalibrationError(
                f"Cannot remove schedule {schedule.name} as {prev_nodes} depend on it."
            )

        self._schedule_dependency.remove_node(sched_idx)

        if sched_key in self._schedules:
            del self._schedules[sched_key]
            del self._schedules_qubits[sched_key]

        # Clean the parameter to schedule mapping.
        self._clean_parameter_map(schedule.name, qubits)

    def _clean_parameter_map(self, schedule_name: str, qubits: Tuple[int, ...]):
        """Clean the parameter to schedule mapping for the given schedule, parameter and qubits.

        Args:
            schedule_name: The name of the schedule.
            qubits: The qubits to which this schedule applies.

        """
        keys_to_remove = []  # of the form (schedule.name, parameter.name, qubits)
        for key in self._parameter_map:
            if key.schedule == schedule_name and key.qubits == qubits:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._parameter_map[key]

            # Key set is a set of tuples (schedule.name, parameter.name, qubits)
            for param, key_set in self._parameter_map_r.items():
                if key in key_set:
                    key_set.remove(key)

        # Remove entries that do not point to at least one (schedule.name, parameter.name, qubits)
        keys_to_delete = []
        for param, key_set in self._parameter_map_r.items():
            if not key_set:
                keys_to_delete.append(param)

        for key in keys_to_delete:
            del self._parameter_map_r[key]

    def _register_parameter(
        self,
        parameter: Parameter,
        qubits: Tuple[int, ...],
        schedule: ScheduleBlock = None,
    ):
        """Registers a parameter for the given schedule.

        This method allows self to determine the parameter instance that corresponds to the given
        schedule name, parameter name and qubits.

        Args:
            parameter: The parameter to register.
            qubits: The qubits for which to register the parameter.
            schedule: The schedule to which this parameter belongs. The schedule can
                be None which allows the calibration to accommodate, e.g. qubit frequencies.
        """
        if parameter not in self._hash_to_counter_map:
            self._hash_to_counter_map[parameter] = self._parameter_counter
            self._parameter_counter += 1

        sched_name = schedule.name if schedule else None
        key = ParameterKey(parameter.name, qubits, sched_name)
        self._parameter_map[key] = parameter
        self._parameter_map_r[parameter].add(key)

    @property
    def parameters(self) -> Dict[Parameter, Set[ParameterKey]]:
        """Return a mapping between parameters and parameter keys.

        Returns a dictionary mapping parameters managed by the calibrations to the schedules and
        qubits and parameter names using the parameters. The values of the dict are sets containing
        the parameter keys. Parameters that are not attached to a schedule will have None in place
        of a schedule name.
        """
        return self._parameter_map_r

    def calibration_parameter(
        self,
        parameter_name: str,
        qubits: Union[int, Tuple[int, ...]] = None,
        schedule_name: str = None,
    ) -> Parameter:
        """Return a parameter given its keys.

        Returns a Parameter object given the triplet parameter_name, qubits and schedule_name
        which uniquely determine the context of a parameter.

        Args:
            parameter_name: Name of the parameter to get.
            qubits: The qubits to which this parameter belongs. If qubits is None then
                the default scope is assumed and the key will be an empty tuple.
            schedule_name: The name of the schedule to which this parameter belongs. A
                parameter may not belong to a schedule in which case None is accepted.

        Returns:
             calibration parameter: The parameter that corresponds to the given arguments.

        Raises:
            CalibrationError: If the desired parameter is not found.
        """
        qubits = self._to_tuple(qubits)

        # 1) Check for qubit specific parameters.
        if ParameterKey(parameter_name, qubits, schedule_name) in self._parameter_map:
            return self._parameter_map[ParameterKey(parameter_name, qubits, schedule_name)]

        # 2) Check for default parameters.
        elif ParameterKey(parameter_name, (), schedule_name) in self._parameter_map:
            return self._parameter_map[ParameterKey(parameter_name, (), schedule_name)]
        else:
            raise CalibrationError(
                f"No parameter for {parameter_name} and schedule {schedule_name} "
                f"and qubits {qubits}. No default value exists."
            )

    def add_parameter_value(
        self,
        value: Union[int, float, complex, ParameterValue],
        param: Union[Parameter, str],
        qubits: Union[int, Tuple[int, ...]] = None,
        schedule: Union[ScheduleBlock, str] = None,
        update_inst_map: bool = True,
    ):
        """Add a parameter value to the stored parameters.

        This parameter value may be applied to several channels, for instance, all
        DRAG pulses may have the same standard deviation.

        Args:
            value: The value of the parameter to add. If an int, float, or complex is given
                then the timestamp of the parameter value will automatically be generated
                and set to the current local time of the user.
            param: The parameter or its name for which to add the measured value.
            qubits: The qubits to which this parameter applies.
            schedule: The schedule or its name for which to add the measured parameter value.
            update_inst_map: Update the instruction schedule map if True (the default).

        Raises:
            CalibrationError: If the schedule name is given but no schedule with that name
                exists.
        """
        qubits = self._to_tuple(qubits)

        if isinstance(value, (int, float, complex)):
            value = ParameterValue(value, datetime.now(timezone.utc).astimezone())

        param_name = param.name if isinstance(param, Parameter) else param
        sched_name = schedule.name if isinstance(schedule, ScheduleBlock) else schedule

        registered_schedules = set(key.schedule for key in self._schedules)

        if sched_name and sched_name not in registered_schedules:
            raise CalibrationError(f"Schedule named {sched_name} was never registered.")

        self._params[ParameterKey(param_name, qubits, sched_name)].append(value)

        # When updating the inst_map we need to
        # a) Update all the schedules that use the parameter to be updated
        # b) Update all schedules that reference the updated schedules under a)
        if update_inst_map and schedule is not None:
            param_obj = self.calibration_parameter(param_name, qubits, sched_name)

            # Take care of a) i.e. update all schedules that use the parameter.
            schedules = set(key.schedule for key in self._parameter_map_r[param_obj])
            keys = set(ScheduleKey(sched, qubits) for sched in schedules)

            # Take care of b) i.e. find all schedules that refer to the schedules that
            # make use of the updated parameter.
            schedules.update(used_in_references(keys, self._schedule_dependency))
            self.update_inst_map(schedules, qubits=qubits)

    def _get_channel_index(self, qubits: Tuple[int, ...], chan: PulseChannel) -> int:
        """Get the index of the parameterized channel.

        The return index is determined from the given qubits and the name of the parameter
        in the channel index. The name of this parameter for control channels must be written
        as chqubit_index1.qubit_index2... followed by an optional $index. For example, the
        following parameter names are valid: 'ch1', 'ch1.0', 'ch30.12', and 'ch1.0$1'.

        Args:
            qubits: The qubits for which we want to obtain the channel index.
            chan: The channel with a parameterized name.

        Returns:
            index: The index of the channel. For example, if qubits=(10, 32) and
                chan is a control channel with parameterized index name 'ch1.0'
                the method returns the control channel corresponding to
                qubits (qubits[1], qubits[0]) which is here the control channel of
                qubits (32, 10).

        Raises:
            CalibrationError:
                - If the number of qubits is incorrect.
                - If the number of inferred ControlChannels is not correct.
                - If ch is not a DriveChannel, MeasureChannel, or ControlChannel.
        """
        if isinstance(chan.index, Parameter):
            if isinstance(
                chan, (DriveChannel, MeasureChannel, AcquireChannel, RegisterSlot, MemorySlot)
            ):
                index = int(chan.index.name[2:].split("$")[0])

                if len(qubits) <= index:
                    raise CalibrationError(f"Not enough qubits given for channel {chan}.")

                return qubits[index]

            # Control channels name example ch1.0$1
            if isinstance(chan, ControlChannel):
                channel_index_parts = chan.index.name[2:].split("$")
                qubit_channels = channel_index_parts[0]

                indices = [int(sub_channel) for sub_channel in qubit_channels.split(".")]
                ch_qubits = tuple(qubits[index] for index in indices)
                chs_ = self._control_channel_map.get(ch_qubits, [])

                control_index = 0
                if len(channel_index_parts) == 2:
                    control_index = int(channel_index_parts[1])

                if len(chs_) <= control_index:
                    raise CalibrationError(
                        f"Control channel index {control_index} not found for qubits {qubits}."
                    )

                return chs_[control_index].index

            raise CalibrationError(
                f"{chan} must be a sub-type of {PulseChannel} or an {AcquireChannel}, "
                f"{RegisterSlot}, or a {MemorySlot}."
            )

        return chan.index

    def get_parameter_value(
        self,
        param: Union[Parameter, str],
        qubits: Union[int, Tuple[int, ...]],
        schedule: Union[ScheduleBlock, str, None] = None,
        valid_only: bool = True,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> Union[int, float, complex]:
        """Retrieves the value of a parameter.

        Parameters may be linked. :meth:`get_parameter_value` does the following steps:

        1. Retrieve the parameter object corresponding to (param, qubits, schedule).
        2. The values of this parameter may be stored under another schedule since
           schedules can share parameters. To deal with this, a list of candidate keys
           is created internally based on the current configuration.
        3. Look for candidate parameter values under the candidate keys.
        4. Filter the candidate parameter values according to their date (up until the
           cutoff_date), validity and calibration group.
        5. Return the most recent parameter.

        Args:
            param: The parameter or the name of the parameter for which to get the parameter value.
            qubits: The qubits for which to get the value of the parameter.
            schedule: The schedule or its name for which to get the parameter value.
            valid_only: Use only parameters marked as valid.
            group: The calibration group from which to draw the parameters.
                If not specified this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.

        Returns:
            value: The value of the parameter.

        Raises:
            CalibrationError: If there is no parameter value for the given parameter name and
                pulse channel.
        """
        qubits = self._to_tuple(qubits)

        # 1) Identify the parameter object.
        param_name = param.name if isinstance(param, Parameter) else param
        sched_name = schedule.name if isinstance(schedule, ScheduleBlock) else schedule

        param = self.calibration_parameter(param_name, qubits, sched_name)

        # 2) Get a list of candidate keys restricted to the qubits of interest.
        candidate_keys = []
        for key in self._parameter_map_r[param]:
            candidate_keys.append(ParameterKey(key.parameter, qubits, key.schedule))

        # 3) Loop though the candidate keys to candidate values
        candidates = []
        for key in candidate_keys:
            if key in self._params:
                candidates += self._params[key]

        # If no candidate parameter values were found look for default parameters
        # i.e. parameters that do not specify a qubit.
        if len(candidates) == 0:
            for key in candidate_keys:
                if ParameterKey(key.parameter, (), key.schedule) in self._params:
                    candidates += self._params[ParameterKey(key.parameter, (), key.schedule)]

        # 4) Filter candidate parameter values.
        if valid_only:
            candidates = [val for val in candidates if val.valid]

        candidates = [val for val in candidates if val.group == group]

        if cutoff_date:
            cutoff_date = cutoff_date.astimezone()
            candidates = [val for val in candidates if val.date_time <= cutoff_date]

        if len(candidates) == 0:
            msg = f"No candidate parameter values for {param_name} in calibration group {group} "

            if qubits:
                msg += f"on qubits {qubits} "

            if sched_name:
                msg += f"in schedule {sched_name} "

            if cutoff_date:
                msg += f"with cutoff date: {cutoff_date}"

            raise CalibrationError(msg)

        # 5) Return the most recent parameter.
        return max(enumerate(candidates), key=lambda x: (x[1].date_time, x[0]))[1].value

    def _standardize_assign_params(
        self,
        assign_params: Dict,
        qubits: Tuple[int, ...],
        schedule_name: str,
    ) -> Dict[ParameterKey, ParameterValueType]:
        """Standardize the format of manually specified parameter assignments.

        Users specify parameter assignment dictionaries as tuples. This function (a) converts
        these tuples to ``ParameterKey`` instances and (b) looks up parameter links between
        schedules and adds them to the standardized returned parameter assignment dictionary.

        Args:
            assign_params: The dictionary that specifies how to assign parameters.
            qubits: The qubits for which to get a schedule.
            schedule_name: The name of the schedule in which the parameters can be found.

        Returns:
            A dict with :class:`.ParameterKey`s as keys and :class:`.ParameterValueType`s as values.
        """
        linked_assign_params = {}

        if assign_params:
            # Add parameter links for automatic linking.
            for param, value in assign_params.items():
                if isinstance(param, str):
                    key = ParameterKey(param, qubits, schedule_name)
                else:
                    key = ParameterKey(*param)
                linked_assign_params[key] = value
                param = self.calibration_parameter(*key)
                for key2 in self._parameter_map_r[param]:  # Loop over linked keys pointing to param
                    # Link only over the same qubits.
                    if key2.qubits == ():
                        key2 = ParameterKey(key2.parameter, key.qubits, key2.schedule)

                    if key2 not in linked_assign_params:
                        linked_assign_params[key2] = value

        return linked_assign_params

    def get_schedule(
        self,
        name: str,
        qubits: Union[int, Tuple[int, ...]],
        assign_params: Dict[Union[str, ParameterKey], ParameterValueType] = None,
        group: Optional[str] = "default",
        cutoff_date: datetime = None,
    ) -> ScheduleBlock:
        """Get the template schedule with parameters assigned to values.

        All the parameters in the template schedule block will be assigned to the values managed
        by the calibrations unless they are specified in assign_params. In this case the value in
        assign_params will override the value stored by the calibrations. A parameter value in
        assign_params may also be a :class:`ParameterExpression`.

        .. code-block:: python

            # Get an xp schedule with a parametric amplitude
            sched = cals.get_schedule("xp", 3, assign_params={"amp": Parameter("amp")})

            # Get an echoed-cross-resonance schedule between qubits (0, 2) where the xp echo gates
            # are referenced schedules but leave their amplitudes as parameters.
            assign_dict = {("amp", (0,), "xp"): Parameter("my_amp")}
            sched = cals.get_schedule("cr", (0, 2), assign_params=assign_dict)

        Args:
            name: The name of the schedule to get.
            qubits: The qubits for which to get the schedule.
            assign_params: The parameters to assign manually. Each parameter is specified by a
                ParameterKey which is a named tuple of the form (parameter name, qubits,
                schedule name). Each entry in assign_params can also be a string corresponding
                to the name of the parameter. In this case, the schedule name and qubits of the
                corresponding ParameterKey will be the name and qubits given as arguments to
                get_schedule.
            group: The calibration group from which to draw the parameters. If not specified
                this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.

        Returns:
            schedule: A copy of the template schedule with all parameters assigned.

        Raises:
            CalibrationError: If the name of the schedule is not known.
            CalibrationError: If a parameter could not be found.
        """
        qubits = self._to_tuple(qubits)

        assign_params = self._standardize_assign_params(assign_params, qubits, name)

        schedule = self.get_template(name, qubits)

        # assign any references using a depth first-search.
        referenced_schedules = {}
        for ref in schedule.references.unassigned():
            ref_name, ref_qubits = reference_info(ref, qubits)
            ref_assign = {}
            for key, val in assign_params.items():
                if key.schedule == ref_name and (key.qubits == tuple() or key.qubits == ref_qubits):
                    ref_assign[key] = val

            referenced_schedules[ref] = self.get_schedule(
                ref_name,
                ref_qubits,
                assign_params=ref_assign,
            )
        schedule = schedule.assign_references(referenced_schedules, inplace=False)

        # Retrieve the channel indices based on the qubits and bind them.
        binding_dict = {}
        for ch in schedule.channels:
            if ch.is_parameterized():
                binding_dict[ch.index] = self._get_channel_index(qubits, ch)

        # Binding the channel indices makes it easier to deal with parameters later on
        schedule = schedule.assign_parameters(binding_dict, inplace=False)

        # Now assign the other parameters
        assigned_schedule = self._assign(schedule, qubits, assign_params, group, cutoff_date)

        free_params = set()
        for param_value in assign_params.values():
            if isinstance(param_value, ParameterExpression):
                free_params.add(param_value)

        if len(assigned_schedule.parameters) != len(free_params):
            raise CalibrationError(
                f"The number of free parameters {len(assigned_schedule.parameters)} in "
                f"the assigned schedule differs from the requested number of free "
                f"parameters {len(free_params)}."
            )

        return assigned_schedule

    def _assign(
        self,
        schedule: ScheduleBlock,
        qubits: Tuple[int, ...],
        assign_params: Dict[Union[str, ParameterKey], ParameterValueType],
        group: Optional[str] = "default",
        cutoff_date: datetime = None,
    ) -> ScheduleBlock:
        """Assign parameters in a schedule.

        This function builds the binding dictionary and ensures that manually specified parameters
        have priority over those managed by the calibrations class.

        Args:
            schedule: The schedule with assigned channel indices for which we wish to
                assign values to non-channel parameters.
            qubits: The qubits for which to get the schedule.
            assign_params: The parameters to manually assign. See get_schedules for details.
            group: The calibration group of the parameters.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.

        Returns:
            ret_schedule: The schedule with assigned parameters.

        Raises:
            CalibrationError: If a channel has not been assigned.
            CalibrationError: If there is an ambiguous parameter assignment.
            CalibrationError: If there are inconsistencies between a called schedule and the
                template schedule registered under the name of the called schedule.
        """
        # 1) Restrict the given qubits to those in the given schedule.
        qubit_set = set()
        for chan in schedule.channels:
            if isinstance(chan.index, ParameterExpression):
                raise CalibrationError(
                    f"Parametric channels must be assigned. {chan} is parametric."
                )

            if isinstance(chan, (DriveChannel, MeasureChannel)):
                qubit_set.add(chan.index)

            if isinstance(chan, ControlChannel):
                for qubit in self._controls_config_r[chan]:
                    qubit_set.add(qubit)

        qubits_ = tuple(qubit for qubit in qubits if qubit in qubit_set)

        # 2) Build the parameter binding dictionary.
        # 2a) Handle parameter assignments that come outside of Calibrations
        binding_dict = {}
        assignment_table = {}
        for key, value in assign_params.items():
            key_orig = key
            if key.qubits == ():
                key = ParameterKey(key.parameter, qubits_, key.schedule)
                if key in assign_params:
                    # if (param, (1,), sched) and (param, (), sched) are both
                    # in assign_params, skip the default value instead of
                    # possibly triggering an error about conflicting
                    # parameters.
                    continue
            elif key.qubits != qubits_:
                continue
            param = self.calibration_parameter(*key)
            if param in schedule.parameters:
                assign_okay = (
                    param not in binding_dict
                    or key.schedule == schedule.name
                    and assignment_table[param].schedule != schedule.name
                )
                if assign_okay:
                    binding_dict[param] = value
                    assignment_table[param] = key_orig
                elif (
                    key.schedule == schedule.name
                    or assignment_table[param].schedule != schedule.name
                ) and binding_dict[param] != value:
                    raise CalibrationError(
                        "Ambiguous assignment: assign_params keys "
                        f"{key_orig} and {assignment_table[param]} "
                        "resolve to the same parameter."
                    )

        # 2b) Handle automatic parameter assignments managed by calibrations.
        if schedule.name in set(key.schedule for key in self._parameter_map):  # skip references
            for param in schedule.parameters:
                key = ParameterKey(param.name, qubits_, schedule.name)
                # Get the parameter object. Since we are dealing with a schedule the name of
                # the schedule is always defined. However, the parameter may be a default
                # parameter for all qubits, i.e. qubits may be an empty tuple.
                param = self.calibration_parameter(*key)

                if param not in binding_dict:
                    binding_dict[param] = self.get_parameter_value(
                        key.parameter,
                        key.qubits,
                        key.schedule,
                        group=group,
                        cutoff_date=cutoff_date,
                    )

        return schedule.assign_parameters(binding_dict, inplace=False)

    def schedules(self) -> List[Dict[str, Any]]:
        """Return the managed schedules in a list of dictionaries.

        Returns:
            data: A list of dictionaries with all the schedules in it. The key-value pairs are

            * ``qubits``: the qubits to which this schedule applies. This may be an empty
                tuple () if the schedule is the default for all qubits.
            * ``schedule``: The schedule.
            * ``parameters``: The parameters in the schedule exposed for convenience.

            This list of dictionaries can easily be converted to a data frame.
        """
        data = []
        for key, sched in self._schedules.items():
            data.append({"qubits": key.qubits, "schedule": sched, "parameters": sched.parameters})

        return data

    def parameters_table(
        self,
        parameters: List[str] = None,
        qubit_list: List[Tuple[int, ...]] = None,
        schedules: List[Union[ScheduleBlock, str]] = None,
        most_recent_only: bool = True,
        group: Optional[str] = None,
    ) -> Dict[str, Union[List[Dict], List[str]]]:
        """A convenience function to help users visualize the values of their parameter.

        Args:
            parameters: The parameter names that should be included in the returned
                table. If None is given then all names are included.
            qubit_list: The qubits that should be included in the returned table.
                If None is given then all channels are returned.
            schedules: The schedules to which to restrict the output.
            most_recent_only: return only the most recent parameter values.
            group: If the group is given then only the parameters from this group are returned.

        Returns:
                A dictionary with the keys "data" and "columns" that can easily
                be converted to a data frame. The "data" are a list of dictionaries
                each holding a parameter value. The "columns" are the keys in the "data"
                dictionaries and are returned in the preferred display order.
        """
        if qubit_list:
            qubit_list = [self._to_tuple(qubits) for qubits in qubit_list]

        # Convert inputs to lists of strings
        if schedules is not None:
            schedules = {sdl.name if isinstance(sdl, ScheduleBlock) else sdl for sdl in schedules}

        # Look for exact matches. Default values will be ignored.
        keys = set()
        for key in self._params.keys():
            if parameters and key.parameter not in parameters:
                continue
            if schedules and key.schedule not in schedules:
                continue
            if qubit_list and key.qubits not in qubit_list:
                continue

            keys.add(key)

        data = []
        if most_recent_only:
            most_recent = {k: max(self._params[k], key=lambda x: x.date_time) for k in keys}

            for key, value in most_recent.items():
                self._append_to_list(data, value, key, group)

        else:
            for key in keys:
                for value in self._params[key]:
                    self._append_to_list(data, value, key, group)

        columns = [
            "parameter",
            "qubits",
            "schedule",
            "value",
            "group",
            "valid",
            "date_time",
            "exp_id",
        ]
        return {"data": data, "columns": columns}

    @staticmethod
    def _append_to_list(
        data: List[Dict], value: ParameterValue, key: ParameterKey, group: Optional[str] = None
    ):
        """Helper function to add a value to the data."""
        if group and value.group != group:
            return

        value_dict = dataclasses.asdict(value)
        value_dict["qubits"] = key.qubits
        value_dict["parameter"] = key.parameter
        value_dict["schedule"] = key.schedule
        value_dict["date_time"] = value_dict["date_time"].strftime("%Y-%m-%d %H:%M:%S.%f%z")
        data.append(value_dict)

    @deprecate_arg(
        name="file_type",
        since="0.6",
        additional_msg="Full calibration saving is now supported in json format. csv is deprecated.",
        package_name="qiskit-experiments",
        predicate=lambda file_type: file_type == "csv",
    )
    def save(
        self,
        file_type: str = "json",
        folder: str = None,
        overwrite: bool = False,
        file_prefix: str = "",
        most_recent_only: bool = False,
    ):
        """Save the parameterized schedules and parameter value.

        .. note::

            Full round-trip serialization of a :class:`.Calibrations` instance
            is only supported in JSON format.
            This may be extended to other file formats in future version.

        Args:
            file_type: The type of file to which to save. By default, this is a json.
                Other file types may be supported in the future.
            folder: The folder in which to save the calibrations.
            overwrite: If the files already exist then they will not be overwritten
                unless overwrite is set to True.
            file_prefix: A prefix to add to the name of the files such as a date tag or a
                UUID.
            most_recent_only: Save only the most recent value. This is set to False by
                default so that when saving to csv all values will be saved.

        Raises:
            CalibrationError: If the files exist and overwrite is not set to True.
        """
        cwd = os.getcwd()
        if folder:
            os.chdir(folder)

        if file_type == "json":
            from .save_utils import calibrations_to_dict

            file_path = file_prefix + ".json"
            if os.path.isfile(file_path) and not overwrite:
                raise CalibrationError(f"{file_path} already exists. Set overwrite to True.")

            canonical_data = calibrations_to_dict(self, most_recent_only=most_recent_only)
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(canonical_data, file, cls=ExperimentEncoder)

        elif file_type == "csv":
            warnings.warn("Schedules are only saved in text format. They cannot be re-loaded.")

            parameter_config_file = file_prefix + "parameter_config.csv"
            parameter_value_file = file_prefix + "parameter_values.csv"
            schedule_file = file_prefix + "schedules.csv"

            if os.path.isfile(parameter_config_file) and not overwrite:
                raise CalibrationError(
                    f"{parameter_config_file} already exists. Set overwrite to True."
                )

            if os.path.isfile(parameter_value_file) and not overwrite:
                raise CalibrationError(
                    f"{parameter_value_file} already exists. Set overwrite to True."
                )

            if os.path.isfile(schedule_file) and not overwrite:
                raise CalibrationError(f"{schedule_file} already exists. Set overwrite to True.")

            # Write the parameter configuration.
            header_keys = ["parameter.name", "parameter unique id", "schedule", "qubits"]
            body = []

            for parameter, keys in self.parameters.items():
                for key in keys:
                    body.append(
                        {
                            "parameter.name": parameter.name,
                            "parameter unique id": self._hash_to_counter_map[parameter],
                            "schedule": key.schedule,
                            "qubits": key.qubits,
                        }
                    )

            with open(parameter_config_file, "w", newline="", encoding="utf-8") as output_file:
                dict_writer = csv.DictWriter(output_file, header_keys)
                dict_writer.writeheader()
                dict_writer.writerows(body)

            # Write the values of the parameters.
            values = self.parameters_table(most_recent_only=most_recent_only)["data"]
            if len(values) > 0:
                header_keys = values[0].keys()

                with open(parameter_value_file, "w", newline="", encoding="utf-8") as output_file:
                    dict_writer = csv.DictWriter(output_file, header_keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(values)

            # Serialize the schedules. For now we just print them.
            header_keys, schedules = self.schedule_information()

            with open(schedule_file, "w", newline="", encoding="utf-8") as output_file:
                dict_writer = csv.DictWriter(output_file, header_keys)
                dict_writer.writeheader()
                dict_writer.writerows(schedules)

        else:
            raise CalibrationError(f"Saving to .{file_type} is not yet supported.")

        os.chdir(cwd)

    @deprecate_func(
        since="0.6",
        additional_msg=(
            "Saving calibration in csv format is deprecate "
            "as well as functions that support this functionality."
        ),
        package_name="qiskit-experiments",
    )
    def schedule_information(self) -> Tuple[List[str], List[Dict]]:
        """Get the information on the schedules stored in the calibrations.

        This function serializes the schedule by simply printing them.

        Returns:
            A tuple, the first element is the header row while the second is a dictionary
            of the schedules in the calibrations where the key is an element of the header
            and the values are the name of the schedule, the qubits to which it applies,
            a string of the schedule.
        """
        # Serialize the schedules. For now we just print them.
        schedules = []
        for key, sched in self._schedules.items():
            schedules.append({"name": key.schedule, "qubits": key.qubits, "schedule": str(sched)})

        return ["name", "qubits", "schedule"], schedules

    @deprecate_func(
        since="0.6",
        additional_msg="Loading and saving calibrations in CSV format is deprecated.",
        package_name="qiskit-experiments",
    )
    def load_parameter_values(self, file_name: str = "parameter_values.csv"):
        """
        Load parameter values from a given file into self._params.

        Args:
            file_name: The name of the file that stores the parameters. Will default to
                parameter_values.csv.
        """
        with open(file_name, encoding="utf-8") as fp:
            reader = csv.DictReader(fp, delimiter=",", quotechar='"')

            for row in reader:
                self._add_parameter_value_from_conf(**row)

        self.update_inst_map()

    def _add_parameter_value_from_conf(
        self,
        value: Union[str, int, float, complex],
        date_time: str,
        valid: Union[str, bool],
        exp_id: str,
        group: str,
        schedule: Union[str, None],
        parameter: str,
        qubits: Union[str, int, Tuple[int, ...]],
    ):
        """Add a parameter value from a parameter configuration.

        The intended usage is :code:`add_parameter_from_conf(**param_conf)`. Entries such
        as ``value`` or ``date_time`` are converted to the proper type.

        Args:
            value: The value of the parameter.
            date_time: The datetime string.
            valid: Whether or not the parameter is valid.
            exp_id: The id of the experiment that created the parameter value.
            group: The calibration group to which the parameter belongs.
            schedule: The schedule to which the parameter belongs. The empty string
                "" is converted to None.
            parameter: The name of the parameter.
            qubits: The qubits on which the parameter acts.
        """
        # TODO remove this after load_parameter_values method is removed.

        param_val = ParameterValue(value, date_time, valid, exp_id, group)

        if schedule == "":
            schedule_name = None
        else:
            schedule_name = schedule

        key = ParameterKey(parameter, self._to_tuple(qubits), schedule_name)
        self.add_parameter_value(param_val, *key, update_inst_map=False)

    @classmethod
    @deprecate_arg(
        name="files",
        new_alias="file_path",
        since="0.6",
        package_name="qiskit-experiments",
    )
    def load(cls, file_path: str) -> "Calibrations":
        """
        Retrieves the parameterized schedules and pulse parameters from the
        given location.

        Args:
            file_path: Path to file location.

        Returns:
            Calibration instance restored from the file.
        """
        from .save_utils import calibrations_from_dict

        with open(file_path, "r", encoding="utf-8") as file:
            # Do we really need branching for data types?
            # Parsing data format and dispatching the loader seems an overkill,
            # but save method intend to support multiple formats.
            cal_data = json.load(file, cls=ExperimentDecoder)

        return calibrations_from_dict(cal_data)

    @staticmethod
    def _to_tuple(qubits: Union[str, int, Tuple[int, ...]]) -> Tuple[int, ...]:
        """Ensure that qubits is a tuple of ints.

        Args:
            qubits: An int, a tuple of ints, or a string representing a tuple of ints.

        Returns:
            qubits: A tuple of ints.

        Raises:
            CalibrationError: If the given input does not conform to an int or
                tuple of ints.
        """
        if qubits is None:
            return tuple()

        if isinstance(qubits, str):
            try:
                return tuple(int(qubit) for qubit in qubits.strip("( )").split(",") if qubit != "")
            except ValueError:
                pass

        if isinstance(qubits, int):
            return (qubits,)

        if isinstance(qubits, list):
            return tuple(qubits)

        if isinstance(qubits, tuple):
            if all(isinstance(n, int) for n in qubits):
                return qubits

        raise CalibrationError(
            f"{qubits} must be int, tuple of ints, or str  that can be parsed"
            f"to a tuple if ints. Received {qubits}."
        )

    def __eq__(self, other: "Calibrations") -> bool:
        """Test equality between two calibrations.

        Two calibration instances are considered equal if
        - The backends have the same name.
        - The backends have the same version.
        - The calibrations contain the same schedules.
        - The stored parameters have the same values.
        """
        if self.backend_name != other.backend_name:
            return False

        if self._backend_version != other.backend_version:
            return False

        # Compare the contents of schedules, schedules are compared by their string
        # representation because they contain parameters.
        for key, schedule in self._schedules.items():
            if repr(schedule) != repr(other._schedules.get(key, None)):
                return False

        # Check the keys.
        if self._schedules.keys() != other._schedules.keys():
            return False

        def _counting(table):
            return Counter(map(lambda d: tuple(d.items()), table["data"]))

        # Use counting sort algorithm to compare unordered sequences
        # https://en.wikipedia.org/wiki/Counting_sort
        return _counting(self.parameters_table()) == _counting(other.parameters_table())

    @deprecate_func(
        since="0.6",
        additional_msg=(
            "Configuration data for Calibrations instance is deprecate. "
            "Please use ExperimentEncoder and ExperimentDecoder to "
            "serialize and deserialize this instance with JSON format."
        ),
        package_name="qiskit-experiments",
    )
    def config(self) -> Dict[str, Any]:
        """Return the settings used to initialize the calibrations.

        Returns:
            The config dictionary of the calibrations instance.

        Raises:
            CalibrationError: If schedules were added outside of the :code:`__init__`
                method. This will remain so until schedules can be serialized.
        """
        if self._has_manually_added_schedule:
            raise CalibrationError(
                f"Config dictionaries for {self.__class__.__name__} are currently "
                "not supported if schedules were added manually."
            )

        kwargs = {
            "coupling_map": self._coupling_map,
            "control_channel_map": ControlChannelMap(self._control_channel_map),
            "libraries": self.libraries,
            "add_parameter_defaults": False,  # the parameters will be added outside of the init
            "backend_name": self._backend_name,
            "backend_version": self._backend_version,
        }

        return {
            "class": self.__class__.__name__,
            "kwargs": kwargs,
            "parameters": self.parameters_table()["data"],
        }

    @classmethod
    @deprecate_func(
        since="0.6",
        additional_msg="This method will be removed and no alternative will be provided.",
        package_name="qiskit-experiments",
    )
    def from_config(cls, config: Dict) -> "Calibrations":
        """Restore Calibration from config data.

        Args:
            config: Configuration data.

        Returns:
            Calibration instance restored from configuration data.
        """
        from .save_utils import calibrations_from_dict

        return calibrations_from_dict(config)

    def __json_encode__(self):
        """Convert to format that can be JSON serialized."""
        from .save_utils import calibrations_to_dict

        return calibrations_to_dict(self, most_recent_only=False)

    @classmethod
    def __json_decode__(cls, value: Dict[str, Any]) -> "Calibrations":
        """Load from JSON compatible format."""
        from .save_utils import calibrations_from_dict

        return calibrations_from_dict(value)
