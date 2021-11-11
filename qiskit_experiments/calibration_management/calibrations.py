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

import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Set, Tuple, Union, List, Optional
import csv
import dataclasses
import warnings
import re

from qiskit.pulse import (
    ScheduleBlock,
    DriveChannel,
    ControlChannel,
    MeasureChannel,
    Call,
    Instruction,
    AcquireChannel,
    RegisterSlot,
    MemorySlot,
    Schedule,
)
from qiskit.pulse.channels import PulseChannel
from qiskit.circuit import Parameter, ParameterExpression
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.database_service.json import serialize_object, deserialize_object
from qiskit_experiments.calibration_management.basis_gate_library import BasisGateLibrary
from qiskit_experiments.calibration_management.parameter_value import ParameterValue
from qiskit_experiments.calibration_management.calibration_key_types import (
    ParameterKey,
    ParameterValueType,
    ScheduleKey,
)


class Calibrations:
    """
    A class to manage schedules with calibrated parameter values. Schedules are
    intended to be fully parameterized, including the index of the channels. See
    the module-level documentation for extra details. Note that only instances of
    ScheduleBlock are supported.
    """

    # The channel indices need to be parameterized following this regex.
    __channel_pattern__ = r"^ch\d[\.\d]*\${0,1}[\d]*$"

    def __init__(
        self,
        control_config: Dict[Tuple[int, ...], List[ControlChannel]] = None,
        library: BasisGateLibrary = None,
        add_parameter_defaults: bool = True,
    ):
        """Initialize the calibrations.

        Calibrations can be initialized from a basis gate library, i.e. a subclass of
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
            control_config: A configuration dictionary of any control channels. The
                keys are tuples of qubits and the values are a list of ControlChannels
                that correspond to the qubits in the keys.
            library: A library class that will be instantiated with the library options to then
                get template schedules to register as well as default parameter values.
            add_parameter_defaults: A boolean to indicate weather the default parameter values of
                the given library should be used to populate the calibrations. By default this
                value is True but can be set to false when deserializing a calibrations object.
        """

        # Mapping between qubits and their control channels.
        self._controls_config = control_config if control_config else {}

        # Store the reverse mapping between control channels and qubits for ease of look-up.
        self._controls_config_r = {}
        for qubits, channels in self._controls_config.items():
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

        # A variable to store all parameter hashes encountered and present them as ordered
        # indices to the user.
        self._hash_to_counter_map = {}
        self._parameter_counter = 0

        self._library = None
        if library is not None:
            self._library = library

            # Add the basis gates
            for gate in library.basis_gates:
                self.add_schedule(library[gate], num_qubits=library.num_qubits(gate))

            # Add the default values
            if add_parameter_defaults:
                for param_conf in library.default_values():
                    schedule_name = param_conf[-1]
                    if schedule_name in library.basis_gates:
                        self.add_parameter_value(*param_conf)

    @property
    def library(self) -> Optional[BasisGateLibrary]:
        """Return the name of the library, e.g. for experiment metadata."""
        return self._library

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
            CalibrationError: If the parameterized channel index is not formatted properly.
            CalibrationError: If several parameters in the same schedule have the same name.
            CalibrationError: If a channel is parameterized by more than one parameter.
            CalibrationError: If the schedule name starts with the prefix of ScheduleBlock.
            CalibrationError: If the schedule calls subroutines that have not been registered.
            CalibrationError: If a :class:`Schedule` is Called instead of a :class:`ScheduleBlock`.
            CalibrationError: If a schedule with the same name exists and acts on a different
                number of qubits.

        """
        qubits = self._to_tuple(qubits)

        if len(qubits) == 0 and num_qubits is None:
            raise CalibrationError("Both qubits and num_qubits cannot simultaneously be None.")

        num_qubits = len(qubits) or num_qubits

        if not isinstance(schedule, ScheduleBlock):
            raise CalibrationError(f"{schedule.name} is not a ScheduleBlock.")

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

        param_indices = set()
        for ch in schedule.channels:
            if isinstance(ch.index, Parameter):
                if len(ch.index.parameters) != 1:
                    raise CalibrationError(f"Channel {ch} can only have one parameter.")

                param_indices.add(ch.index)
                if re.compile(self.__channel_pattern__).match(ch.index.name) is None:
                    raise CalibrationError(
                        f"Parameterized channel must correspond to {self.__channel_pattern__}"
                    )

        # Check that subroutines are present.
        for block in schedule.blocks:
            if isinstance(block, Call):
                if isinstance(block.subroutine, Schedule):
                    raise CalibrationError(
                        "Calling a Schedule is forbidden, call ScheduleBlock instead."
                    )

                if (block.subroutine.name, qubits) not in self._schedules:
                    raise CalibrationError(
                        f"Cannot register schedule block {schedule.name} with unregistered "
                        f"subroutine {block.subroutine.name}."
                    )

        # Clean the parameter to schedule mapping. This is needed if we overwrite a schedule.
        self._clean_parameter_map(schedule.name, qubits)

        # Add the schedule.
        self._schedules[sched_key] = schedule
        self._schedules_qubits[sched_key] = num_qubits

        # Register parameters that are not indices.
        # Do not register parameters that are in call instructions.
        params_to_register = set()
        for inst in self._exclude_calls(schedule, []):
            for param in inst.parameters:
                if param not in param_indices:
                    params_to_register.add(param)

        if len(params_to_register) != len(set(param.name for param in params_to_register)):
            raise CalibrationError(f"Parameter names in {schedule.name} must be unique.")

        for param in params_to_register:
            self._register_parameter(param, qubits, schedule)

    def _exclude_calls(
        self, schedule: ScheduleBlock, instructions: List[Instruction]
    ) -> List[Instruction]:
        """Return the non-Call instructions.

        Recursive function to get all non-Call instructions. This will flatten all blocks
        in a :class:`ScheduleBlock` and return the instructions of the ScheduleBlock leaving
        out any Call instructions.

        Args:
            schedule: A :class:`ScheduleBlock` from which to extract the instructions.
            instructions: The list of instructions that is recursively populated.

        Returns:
            The list of instructions to which all non-Call instructions have been added.
        """
        for block in schedule.blocks:
            if isinstance(block, ScheduleBlock):
                instructions = self._exclude_calls(block, instructions)
            else:
                if not isinstance(block, Call):
                    instructions.append(block)

        return instructions

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
            CalibrationError: if no template schedule for the given schedule name and qubits
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
        """
        qubits = self._to_tuple(qubits)

        sched_key = ScheduleKey(schedule.name, qubits)
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
                chs_ = self._controls_config[ch_qubits]

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
            # are Called schedules but leave their amplitudes as parameters.
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
            schedule: A copy of the template schedule with all parameters assigned
            except for those specified by assign_params.

        Raises:
            CalibrationError: If the name of the schedule is not known.
            CalibrationError: If a parameter could not be found.
        """
        qubits = self._to_tuple(qubits)

        # Standardize the input in the assignment dictionary
        if assign_params:
            assign_params_ = dict()
            for assign_param, value in assign_params.items():
                if isinstance(assign_param, str):
                    assign_params_[ParameterKey(assign_param, qubits, name)] = value
                else:
                    assign_params_[ParameterKey(*assign_param)] = value

            assign_params = assign_params_
        else:
            assign_params = dict()

        # Get the template schedule
        if (name, qubits) in self._schedules:
            schedule = self._schedules[ScheduleKey(name, qubits)]
        elif (name, ()) in self._schedules:
            schedule = self._schedules[ScheduleKey(name, ())]
        else:
            raise CalibrationError(f"Schedule {name} is not defined for qubits {qubits}.")

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
        for param in assign_params.values():
            if isinstance(param, ParameterExpression):
                free_params.add(param)

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
        """Recursively assign parameters in a schedule.

        The recursive behaviour is needed to handle Call instructions as the name of
        the called instruction defines the scope of the parameter. Each time a Call
        is found _assign recurses on the channel-assigned subroutine of the Call
        instruction and the qubits that are in said subroutine. This requires a
        careful extraction of the qubits from the subroutine and in the appropriate
        order. Next, the parameters are identified and assigned. This is needed to
        handle situations where the same parameterized schedule is called but on
        different channels. For example,

        .. code-block:: python

            ch0 = Parameter("ch0")
            ch1 = Parameter("ch1")

            with pulse.build(name="xp") as xp:
                pulse.play(Gaussian(duration, amp, sigma), DriveChannel(ch0))

            with pulse.build(name="xt_xp") as xt:
                pulse.call(xp)
                pulse.call(xp, value_dict={ch0: ch1})

        Here, we define the xp :class:`ScheduleBlock` for all qubits as a Gaussian. Next, we define
        a schedule where both xp schedules are called simultaneously on different channels. We now
        explain a subtlety related to manually assigning values in the case above. In the schedule
        above, the parameters of the Gaussian pulses are coupled, e.g. the xp pulse on ch0 and ch1
        share the same instance of :class:`ParameterExpression`. Suppose now that both pulses have
        a duration and sigma of 160 and 40 samples, respectively, and that the amplitudes are 0.5
        and 0.3 for qubits 0 and 2, respectively. These values are stored in self._params. When
        retrieving a schedule without specifying assign_params, i.e.

        .. code-block:: python

            cals.get_schedule("xt_xp", (0, 2))

        we will obtain the expected schedule with amplitudes 0.5 and 0.3. When specifying the
        following :code:`assign_params = {("amp", (0,), "xp"): Parameter("my_new_amp")}` we
        will obtain a schedule where the amplitudes of the xp pulse on qubit 0 is set to
        :code:`Parameter("my_new_amp")`. The amplitude of the xp pulse on qubit 2 is set to
        the value stored by the calibrations, i.e. 0.3.

        .. code-bloc:: python

            cals.get_schedule(
                "xt_xp",
                (0, 2),
                assign_params = {("amp", (0,), "xp"): Parameter("my_new_amp")}
            )

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
            CalibrationError:
                - If a channel has not been assigned.
                - If there is an ambiguous parameter assignment.
                - If there are inconsistencies between a called schedule and the template
                  schedule registered under the name of the called schedule.
        """
        # 1) Restrict the given qubits to those in the given schedule.
        qubit_set = set()
        for chan in schedule.channels:
            if isinstance(chan.index, ParameterExpression):
                raise (
                    CalibrationError(
                        f"All parametric channels must be assigned before searching for "
                        f"non-channel parameters. {chan} is parametric."
                    )
                )
            if isinstance(chan, (DriveChannel, MeasureChannel)):
                qubit_set.add(chan.index)

            if isinstance(chan, ControlChannel):
                for qubit in self._controls_config_r[chan]:
                    qubit_set.add(qubit)

        qubits_ = tuple(qubit for qubit in qubits if qubit in qubit_set)

        # 2) Recursively assign the parameters in the called instructions.
        ret_schedule = ScheduleBlock(
            alignment_context=schedule.alignment_context,
            name=schedule.name,
            metadata=schedule.metadata,
        )

        for inst in schedule.blocks:
            if isinstance(inst, Call):
                # Check that there are no inconsistencies with the called subroutines.
                template_subroutine = self.get_template(inst.subroutine.name, qubits_)
                if inst.subroutine != template_subroutine:
                    raise CalibrationError(
                        f"The subroutine {inst.subroutine.name} called by {inst.name} does not "
                        f"match the template schedule stored under {template_subroutine.name}."
                    )

                inst = inst.assigned_subroutine()

            if isinstance(inst, ScheduleBlock):
                inst = self._assign(inst, qubits_, assign_params, group, cutoff_date)

            ret_schedule.append(inst, inplace=True)

        # 3) Get the parameter keys of the remaining instructions. At this point in
        #    _assign all parameters in Call instructions that are supposed to be
        #     assigned have been assigned.
        keys = set()

        if ret_schedule.name in set(key.schedule for key in self._parameter_map):
            for param in ret_schedule.parameters:
                keys.add(ParameterKey(param.name, qubits_, ret_schedule.name))

        # 4) Build the parameter binding dictionary.
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
            if param in ret_schedule.parameters:
                assign_okay = (
                    param not in binding_dict
                    or key.schedule == ret_schedule.name
                    and assignment_table[param].schedule != ret_schedule.name
                )
                if assign_okay:
                    binding_dict[param] = value
                    assignment_table[param] = key_orig
                elif (
                    key.schedule == ret_schedule.name
                    or assignment_table[param].schedule != ret_schedule.name
                ) and binding_dict[param] != value:
                    raise CalibrationError(
                        "Ambiguous assignment: assign_params keys "
                        f"{key_orig} and {assignment_table[param]} "
                        "resolve to the same parameter."
                    )

        for key in keys:
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

        return ret_schedule.assign_parameters(binding_dict, inplace=False)

    def schedules(self) -> List[Dict[str, Any]]:
        """Return the managed schedules in a list of dictionaries.

        Returns:
            data: A list of dictionaries with all the schedules in it. The key-value pairs are

                * 'qubits': the qubits to which this schedule applies. This may be an empty
                  tuple () if the schedule is the default for all qubits.
                * 'schedule': The schedule.
                * 'parameters': The parameters in the schedule exposed for convenience.

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
    ) -> Dict[str, Union[List[Dict], List[str]]]:
        """A convenience function to help users visualize the values of their parameter.

        Args:
            parameters: The parameter names that should be included in the returned
                table. If None is given then all names are included.
            qubit_list: The qubits that should be included in the returned table.
                If None is given then all channels are returned.
            schedules: The schedules to which to restrict the output.

        Returns:
                A dictionary with the keys "data" and "columns" that can easily
                be converted to a data frame. The "data" are a list of dictionaries
                each holding a parameter value. The "columns" are the keys in the "data"
                dictionaries and are returned in the preferred display order.
        """
        if qubit_list:
            qubit_list = [self._to_tuple(qubits) for qubits in qubit_list]

        data = []

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

        for key in keys:
            for value in self._params[key]:
                value_dict = dataclasses.asdict(value)
                value_dict["qubits"] = key.qubits
                value_dict["parameter"] = key.parameter
                value_dict["schedule"] = key.schedule
                value_dict["date_time"] = value_dict["date_time"].strftime("%Y-%m-%d %H:%M:%S.%f%z")
                data.append(value_dict)

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

    def save(
        self,
        file_type: str = "csv",
        folder: str = None,
        overwrite: bool = False,
        file_prefix: str = "",
    ):
        """Save the parameterized schedules and parameter value.

        The schedules and parameter values can be stored in csv files. This method creates
        three files:

        * parameter_config.csv: This file stores a table of parameters which indicates
          which parameters appear in which schedules.
        * parameter_values.csv: This file stores the values of the calibrated parameters.
        * schedules.csv: This file stores the parameterized schedules.

        Warning:
            Schedule blocks will only be saved in string format and can therefore not be
            reloaded and must instead be rebuilt.

        Args:
            file_type: The type of file to which to save. By default this is a csv.
                Other file types may be supported in the future.
            folder: The folder in which to save the calibrations.
            overwrite: If the files already exist then they will not be overwritten
                unless overwrite is set to True.
            file_prefix: A prefix to add to the name of the files such as a date tag or a
                UUID.

        Raises:
            CalibrationError: if the files exist and overwrite is not set to True.
        """
        warnings.warn("Schedules are only saved in text format. They cannot be re-loaded.")

        cwd = os.getcwd()
        if folder:
            os.chdir(folder)

        parameter_config_file = file_prefix + "parameter_config.csv"
        parameter_value_file = file_prefix + "parameter_values.csv"
        schedule_file = file_prefix + "schedules.csv"

        if os.path.isfile(parameter_config_file) and not overwrite:
            raise CalibrationError(
                f"{parameter_config_file} already exists. Set overwrite to True."
            )

        if os.path.isfile(parameter_value_file) and not overwrite:
            raise CalibrationError(f"{parameter_value_file} already exists. Set overwrite to True.")

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

        if file_type == "csv":
            with open(parameter_config_file, "w", newline="", encoding="utf-8") as output_file:
                dict_writer = csv.DictWriter(output_file, header_keys)
                dict_writer.writeheader()
                dict_writer.writerows(body)

            # Write the values of the parameters.
            values = self.parameters_table()["data"]
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
                self.add_parameter_value_from_conf(**row)

    def add_parameter_value_from_conf(
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
        param_val = ParameterValue(value, date_time, valid, exp_id, group)

        if schedule == "":
            schedule_name = None
        else:
            schedule_name = schedule

        key = ParameterKey(parameter, self._to_tuple(qubits), schedule_name)
        self.add_parameter_value(param_val, *key)

    @classmethod
    def load(cls, files: List[str]) -> "Calibrations":
        """
        Retrieves the parameterized schedules and pulse parameters from the
        given location.
        """
        raise CalibrationError("Full calibration loading is not implemented yet.")

    def serialize(self, save_parameters: bool = True) -> Dict:
        """Serializes the class to a Dictionary.

        Args:
            save_parameters: If set to True, the default value, then all the values of the
                calibrations will also be serialized.

        Returns:
            A dict object that represents the calibrations and can be used to rebuild the
            calibrations. See :meth:`deserialize`.
        """

        if self._library is None:
            raise CalibrationError(
                "Cannot serialize calibrations that are not constructed from a library."
            )

        # encode the configuration of the controls.
        serialized_controls_config = {}
        for qubits, channels in self._controls_config.items():
            serialized_controls_config[qubits] = [chan.index for chan in channels]

        serialized_cals = serialize_object(type(self))
        serialized_cals["__value__"]["__library__"] = self._library.serialize()
        serialized_cals["__value__"]["__controls_config__"] = serialized_controls_config

        if save_parameters:
            serialized_cals["__value__"]["__parameter_values__"] = self.parameters_table()["data"]

        return serialized_cals

    @classmethod
    def deserialize(cls, serialized_dict: Dict, *args):
        """Deserialize from a dictionary.

        Args:
            serialized_dict: The dictionary from which to create the calibrations instance.

        Returns:
            An instance of Calibrations.
        """

        # Deserialize the library.
        library = BasisGateLibrary.deserialize(serialized_dict["__value__"].pop("__library__"))

        # Deserialize the control channel configuration.
        control_config = {}
        for qubits, channels in serialized_dict["__value__"]["__controls_config__"].items():
            control_config[qubits] = [ControlChannel(index) for index in channels]

        params = serialized_dict["__value__"].get("__parameter_values__", [])
        add_library_params = True if len(params) == 0 else False

        cals = deserialize_object(
            serialized_dict["__value__"]["__module__"],
            serialized_dict["__value__"]["__name__"],
            tuple(),
            {
                "library": library,
                "control_config": control_config,
                "add_parameter_defaults": add_library_params,
            },
        )

        # Add the parameter values if any
        params = serialized_dict["__value__"].get("__parameter_values__", [])
        for param_conf in params:
            cals.add_parameter_value_from_conf(**param_conf)

        return cals

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
