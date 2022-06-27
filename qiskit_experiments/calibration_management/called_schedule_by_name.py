# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class to call schedule by name."""

from typing import Optional, Tuple, Union

from qiskit.pulse import ScheduleBlock
from qiskit.pulse.channels import Channel
from qiskit.pulse.transforms import AlignmentKind


class CalledScheduleByName(ScheduleBlock):
    """A schedule block to reference another schedule by name.

    This class allows us to uncouple a template library from external dependencies.
    Note that schedules in a library may consist of multiple pulses which 
    is not defined within the current scope.
    The typical example is an echoed-cross-resonance schedule where the schedule

    of the echoed x-gate is called. Instead of using the ``pulse.Call`` instruction
    we can call the schedule of the x-pulse by name using ``CalledScheduleByName``.
    By doing so we do not need to specify the schedule of the x-pulse. This will be resolved
    by the instance of :class:`.Calibrations` by the :meth:`get_schedule` method.
    """

    def __init__(
        self,
        schedule_name: str,
        channels: Union[Channel, Tuple[Channel]],
        alignment_context: Optional[AlignmentKind] = None,
    ):
        """
        Args:
            schedule_name: The name of the schedule that will be called.
            channels: The channels to which the called schedule applies.
            qubits: The qubits to which the called schedule applies to.
            alignment_context: The instance that manages the alignment of the schedules.
        """
        super().__init__(name=schedule_name, alignment_context=alignment_context)

        if not isinstance(channels, tuple):
            channels = (channels,)

        self._channels = channels
        for channel in self.channels:
            self._parameter_manager.update_parameter_table(channel)

    @property
    def channels(self) -> Tuple[Channel]:
        """Return the channels of the called schedule."""
        return self._channels

    def is_schedulable(self) -> bool:
        """A schedule called by name cannot be scheduled."""
        return False

    def __eq__(self, other):
        """Check only names and channels."""
        # 0. type check
        if not isinstance(other, type(self)):
            return False

        # 1. transformation check
        if self.alignment_context != other.alignment_context:
            return False

        # 2. channel check
        if set(self.channels) != set(other.channels):
            return False

        # 3. size check
        if len(self) != len(other):
            return False

        return True

    def append(
        self, block: "BlockComponent", name: Optional[str] = None, inplace: bool = True
    ) -> "ScheduleBlock":
        """One cannot append to a schedule called by name."""
        raise NotImplementedError(f"Cannot append to a {self.__class__.__name__}.")
