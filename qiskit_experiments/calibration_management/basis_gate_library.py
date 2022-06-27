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

"""
A collections of libraries to setup Calibrations.

Note that the set of available libraries will be extended in future releases.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set
from warnings import warn

from qiskit.circuit import Parameter
import qiskit.pulse as pulse
from qiskit.pulse import ScheduleBlock
from qiskit.pulse.transforms import AlignSequential

from qiskit_experiments.calibration_management.calibration_key_types import DefaultCalValue
from qiskit_experiments.calibration_management.called_schedule_by_name import CalledScheduleByName
from qiskit_experiments.exceptions import CalibrationError


class BasisGateLibrary(ABC, Mapping):
    """A base class for libraries of basis gates to make it easier to setup Calibrations."""

    # Location where default parameter values are stored. These may be updated at construction.
    __default_values__ = dict()

    def __init__(
        self,
        basis_gates: Optional[List[str]] = None,
        default_values: Optional[Dict] = None,
        **extra_kwargs,
    ):
        """Setup the library.

        Args:
            basis_gates: The basis gates to generate.
            default_values: A dictionary to override library default parameter values.
            extra_kwargs: Extra key-word arguments of the subclasses that are saved to be able
                to reconstruct the library using the :meth:`__init__` method.

        Raises:
            CalibrationError: If on of the given basis gates is not supported by the library.
        """
        # Update the default values.
        self._extra_kwargs = extra_kwargs
        self._default_values = self.__default_values__.copy()
        if default_values is not None:
            self._default_values.update(default_values)

        if basis_gates is None:
            basis_gates = list(self.__supported_gates__)

        for gate in basis_gates:
            if gate not in self.__supported_gates__:
                raise CalibrationError(
                    f"Gate {gate} is not supported by {self.__class__.__name__}. "
                    f"Supported gates are: {self.__supported_gates__}."
                )

        self._schedules = self._build_schedules(set(basis_gates))

    @property
    @abstractmethod
    def __supported_gates__(self) -> Dict[str, int]:
        """Return the supported gates of the library.

        The key is the name of the gate and the value is the number of qubits it applies to.
        """
        raise NotImplementedError

    def __getitem__(self, name: str) -> ScheduleBlock:
        """Return the schedule."""
        if name not in self._schedules:
            raise CalibrationError(f"Gate {name} is not contained in {self.__class__.__name__}.")

        return self._schedules[name]

    def __contains__(self, name: str) -> bool:
        """Check if the basis gate is in the library."""
        return name in self._schedules

    def __hash__(self) -> int:
        """Return the hash of the library by computing the hash of the schedule strings."""
        data_to_hash = []
        for name, schedule in sorted(self._schedules.items()):
            data_to_hash.append((name, str(schedule), self.__supported_gates__[name]))

        return hash(tuple(data_to_hash))

    def __len__(self):
        """The length of the library defined as the number of basis gates."""
        return len(self._schedules)

    def __iter__(self):
        """Return an iterator over the basis gate library."""
        return iter(self._schedules)

    def num_qubits(self, name: str) -> int:
        """Return the number of qubits that the schedule with the given name acts on."""
        return self.__supported_gates__[name]

    @property
    def basis_gates(self) -> List[str]:
        """Return the basis gates supported by the library."""
        return list(self._schedules)

    def default_values(self) -> List[DefaultCalValue]:
        """Return the default values for the parameters.

        Returns
            A list of tuples is returned. These tuples are structured so that instances of
            :class:`Calibrations` can call :meth:`add_parameter_value` on the tuples.
        """
        defaults = []
        for name, schedule in self.items():
            for param in schedule.parameters:
                if "ch" not in param.name:
                    value = self._default_values.get(param.name, None)
                    if value is not None:
                        defaults.append(DefaultCalValue(value, param.name, tuple(), name))

        return defaults

    @abstractmethod
    def _build_schedules(self, basis_gates: Set[str]) -> Dict[str, ScheduleBlock]:
        """Build the schedules stored in the library.

        This method is called as the last step in the :meth:`__init__`. Subclasses must implement
        :meth:`_build_schedules` to build the schedules of the library based on the inputs given
        to the :meth:`__init__` method.

        Args:
            basis_gates: The set of basis gates to build.

        Returns:
            A dictionary where the keys are the names of the schedules/basis gates and the values
            are the corresponding schedules.
        """

    def config(self) -> Dict[str, Any]:
        """Return the settings used to initialize the library."""

        kwargs = {"basis_gates": self.basis_gates, "default_values": self._default_values}
        kwargs.update(self._extra_kwargs)

        return {
            "class": self.__class__.__name__,
            "kwargs": kwargs,
            "hash": self.__hash__(),
        }

    @classmethod
    def from_config(cls, config: Dict) -> "BasisGateLibrary":
        """Deserialize the library given the input dictionary"""
        library = cls(**config["kwargs"])

        if hash(library) != config["hash"]:
            warn(
                "Deserialized basis gate library's hash does not match the hash of the serialized "
                "library. Typically, the hash changes when the internal structure of the template "
                "schedules has been changed."
            )

        return library

    def __json_encode__(self):
        """Convert to format that can be JSON serialized."""
        return self.config()

    @classmethod
    def __json_decode__(cls, value: Dict[str, Any]) -> "BasisGateLibrary":
        """Load from JSON compatible format."""
        return cls.from_config(value)


class FixedFrequencyTransmon(BasisGateLibrary):
    """A library of gates for fixed-frequency superconducting qubit architectures.

    Note that for now this library supports single-qubit gates and will be extended
    in the future.
    """

    __default_values__ = {"duration": 160, "amp": 0.5, "β": 0.0}

    def __init__(
        self,
        basis_gates: Optional[List[str]] = None,
        default_values: Optional[Dict] = None,
        link_parameters: bool = True,
    ):
        """Setup the schedules.

        Args:
            basis_gates: The basis gates to generate.
            default_values: Default values for the parameters this dictionary can contain
                the following keys: "duration", "amp", "β", and "σ". If "σ" is not provided
                this library will take one fourth of the pulse duration as default value.
            link_parameters: if set to True then the amplitude and DRAG parameters of the
                X and Y gates will be linked as well as those of the SX and SY gates.
        """
        self._link_parameters = link_parameters

        extra_kwargs = {"link_parameters": link_parameters}

        super().__init__(basis_gates, default_values, **extra_kwargs)

    @property
    def __supported_gates__(self) -> Dict[str, int]:
        """The gates that this library supports."""
        return {"x": 1, "y": 1, "sx": 1, "sy": 1}

    def _build_schedules(self, basis_gates: Set[str]) -> Dict[str, ScheduleBlock]:
        """Build the schedule of the class."""
        dur = Parameter("duration")
        sigma = Parameter("σ")

        x_amp, x_beta = Parameter("amp"), Parameter("β")

        if self._link_parameters:
            y_amp, y_beta = 1.0j * x_amp, x_beta
        else:
            y_amp, y_beta = Parameter("amp"), Parameter("β")

        sx_amp, sx_beta = Parameter("amp"), Parameter("β")

        if self._link_parameters:
            sy_amp, sy_beta = 1.0j * sx_amp, sx_beta
        else:
            sy_amp, sy_beta = Parameter("amp"), Parameter("β")

        # Create the schedules for the gates
        sched_x = self._single_qubit_schedule("x", dur, x_amp, sigma, x_beta)
        sched_y = self._single_qubit_schedule("y", dur, y_amp, sigma, y_beta)
        sched_sx = self._single_qubit_schedule("sx", dur, sx_amp, sigma, sx_beta)
        sched_sy = self._single_qubit_schedule("sy", dur, sy_amp, sigma, sy_beta)

        schedules = dict()
        for sched in [sched_x, sched_y, sched_sx, sched_sy]:
            if sched.name in basis_gates:
                schedules[sched.name] = sched

        return schedules

    @staticmethod
    def _single_qubit_schedule(
        name: str,
        dur: Parameter,
        amp: Parameter,
        sigma: Parameter,
        beta: Parameter,
    ) -> ScheduleBlock:
        """Build a single qubit pulse."""

        chan = pulse.DriveChannel(Parameter("ch0"))

        with pulse.build(name=name) as sched:
            pulse.play(pulse.Drag(duration=dur, amp=amp, sigma=sigma, beta=beta), chan)

        return sched

    def default_values(self) -> List[DefaultCalValue]:
        """Return the default values for the parameters.

        This overrides the method of the base class to account for parameter linkages and
        standard sigma to duration relations.

        Returns
            A list of tuples is returned. These tuples are structured so that instances of
            :class:`Calibrations` can call :meth:`add_parameter_value` on the tuples.
        """
        defaults = []
        for name, schedule in self.items():
            for param in schedule.parameters:
                if "ch" not in param.name:
                    if "y" in name and self._link_parameters:
                        continue

                    if param.name == "σ" and "σ" not in self._default_values:
                        value = self._default_values["duration"] / 4
                    else:
                        value = self._default_values[param.name]

                    if name in {"sx", "sy"} and param.name == "amp":
                        value /= 2.0

                    if "y" in name and param.name == "amp":
                        value *= 1.0j

                    defaults.append(DefaultCalValue(value, param.name, tuple(), name))

        return defaults


class EchoedCrossResonance(BasisGateLibrary):
    """A gate library to define the echoed cross-resonance gate."""

    __default_values__ = {
        "duration": 640,
        "width": 384,
        "σ": 64,
        "amp": 0.5,
        "amp_trg": 0.0,
    }

    def __init__(
        self,
        pulse_on_target: bool = True,
        basis_gates: Optional[List[str]] = None,
        default_values: Optional[Dict] = None,
    ):
        """Setup the schedules.

        Args:
            pulse_on_target: A boolean to indicate if rotary/cancellation tones are
                used in the CR gate. If this value is set to False then no pulse is played
                on the target qubit during the cross-resonance pulse.
            basis_gates: The basis gates to generate.
            default_values: Default values for the parameters. This dictionary can contain
                the following keys: "duration", "amp", "amp_trg", "width", and "σ".
        """
        self._pulse_on_target = pulse_on_target
        super().__init__(basis_gates, default_values)

    @property
    def __supported_gates__(self) -> Dict[str, int]:
        """The gates that this library supports and the number of qubits they apply to."""
        return {"ecr": 2, "cr45p": 2, "cr45m": 2}

    def _build_schedules(self, basis_gates: Set[str]) -> Dict[str, ScheduleBlock]:
        """Build the schedules of the library."""

        control = pulse.DriveChannel(Parameter("ch0"))
        target = pulse.DriveChannel(Parameter("ch1"))
        cr_chan = pulse.ControlChannel(Parameter("ch0.1"))

        cr_amp = Parameter("amp")
        trg_amp = Parameter("amp_trg")
        cr_sigma = Parameter("σ")
        cr_duration = Parameter("duration")
        cr_width = Parameter("width")

        cr45p = pulse.GaussianSquare(
            duration=cr_duration, amp=cr_amp, sigma=cr_sigma, width=cr_width, name="cr45p"
        )

        cr45m = pulse.GaussianSquare(
            duration=cr_duration, amp=-cr_amp, sigma=cr_sigma, width=cr_width, name="cr45m"
        )

        cr45p_target = pulse.GaussianSquare(
            duration=cr_duration, amp=trg_amp, sigma=cr_sigma, width=cr_width, name="cr45p_target"
        )

        cr45m_target = pulse.GaussianSquare(
            duration=cr_duration, amp=-trg_amp, sigma=cr_sigma, width=cr_width, name="cr45m_target"
        )

        with pulse.build(name="cr45p") as cr45p_sched:
            pulse.play(cr45p, cr_chan)
            if self._pulse_on_target:
                pulse.play(cr45p_target, target)

        with pulse.build(name="cr45m") as cr45m_sched:
            pulse.play(cr45m, cr_chan)
            if self._pulse_on_target:
                pulse.play(cr45m_target, target)

        ecr_sched = pulse.ScheduleBlock(name="ecr", alignment_context=AlignSequential())
        ecr_sched.append(cr45p_sched)
        ecr_sched.append(CalledScheduleByName("x", control))
        ecr_sched.append(cr45m_sched)
        ecr_sched.append(CalledScheduleByName("x", control))

        return {"ecr": ecr_sched, "cr45m": cr45m_sched, "cr45p": cr45p_sched}
