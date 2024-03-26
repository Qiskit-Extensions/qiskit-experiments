# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base calibrations class."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.calibration_management.parameter_value import ParameterValue
from qiskit_experiments.calibration_management.calibration_key_types import (
    ParameterKey,
    ParameterValueType,
)


class BaseCalibrations(ABC):
    """An abstract base calibration class that defines the methods needed by cal. experiments.

    A calibration experiment uses an instance of `BaseCalibrations` that defines where
    to get parameter values from and where to save them. This class also defines a method from
    which to retrieve pulse-schedules.
    """

    @abstractmethod
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
        """

    @abstractmethod
    def get_parameter_value(
        self,
        param: Union[Parameter, str],
        qubits: Union[int, Tuple[int, ...]],
        schedule: Union[ScheduleBlock, str, None] = None,
    ) -> Union[int, float, complex]:
        """Retrieves the value of a parameter.

        Parameters may be linked. :meth:`get_parameter_value` does the following steps:

        Args:
            param: The parameter or the name of the parameter for which to get the parameter value.
            qubits: The qubits for which to get the value of the parameter.
            schedule: The schedule or its name for which to get the parameter value.

        Returns:
            value: The value of the parameter.
        """

    @abstractmethod
    def get_schedule(
        self,
        name: str,
        qubits: Union[int, Tuple[int, ...]],
        assign_params: Dict[Union[str, ParameterKey], ParameterValueType] = None,
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

        Returns:
            schedule: A copy of the template schedule with all parameters assigned.
        """
