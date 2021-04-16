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

"""Different data analysis steps."""

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

from qiskit_experiments.data_processing.data_action import DataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError


class IQPart(DataAction):
    """Abstract class for IQ data post-processing."""

    def __init__(self, scale: Optional[float] = None, validate: bool = True):
        """
        Args:
            scale: Float with which to multiply the IQ data.
            validate: If set to False the DataAction will not validate its input.
        """
        self.scale = scale
        super().__init__(validate)

    @abstractmethod
    def _process(self, datum: np.array) -> np.array:
        """Defines how the IQ point will be processed.

        Args:
            datum: A 3D array of shots, qubits, and a complex IQ point as [real, imaginary].

        Returns:
            Processed IQ point.
        """

    def _format_data(self, datum: Any, validate: bool = True) -> Any:
        """Check that the IQ data has the correct format and convert to numpy array.

        Args:
            datum: A single item of data which corresponds to single-shot IQ data. It should
                have dimension three: shots, qubits, iq-point as [real, imaginary].
            validate: If True the DataAction checks that the format of the datum is valid.

        Returns:
            datum as a numpy array.

        Raises:
            DataProcessorError: If the datum does not have the correct format.
        """
        datum = np.asarray(datum, dtype=complex)

        if validate and len(datum.shape) != 3:
            raise DataProcessorError(
                f"Single-shot data given {self.__class__.__name__}"
                f"must be a 3D array. Instead, a {len(datum.shape)}D "
                f"array was given."
            )

        return datum

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}(validate: {self._validate}, scale: {self.scale})"


class ToReal(IQPart):
    """IQ data post-processing. Isolate the real part of the IQ data."""

    def _process(self, datum: np.array) -> np.array:
        """Take the real part of the IQ data.

        Args:
            datum: A 3D array of shots, qubits, and a complex IQ point as [real, imaginary].

        Returns:
            A 2D array of shots, qubits. Each entry is the real part of the given IQ data.
        """
        if self.scale is None:
            return datum[:, :, 0]

        return datum[:, :, 0] * self.scale


class ToImag(IQPart):
    """IQ data post-processing. Isolate the imaginary part of the IQ data."""

    def _process(self, datum: np.array) -> np.array:
        """Take the imaginary part of the IQ data.

        Args:
            datum: A 3D array of shots, qubits, and a complex IQ point as [real, imaginary].

        Returns:
            A 2D array of shots, qubits. Each entry is the imaginary part of the given IQ data.
        """
        if self.scale is None:
            return datum[:, :, 1]

        return datum[:, :, 1] * self.scale


class Probability(DataAction):
    """Count data post processing. This returns qubit 1 state probabilities."""

    def __init__(self, outcome: str, validate: bool = True):
        """Initialize a counts to probability data conversion.

        Args:
            outcome: The bitstring for which to compute the probability.
            validate: If set to False the DataAction will not validate its input.
        """
        self._outcome = outcome
        super().__init__(validate)

    def _format_data(self, datum: dict, validate: bool = True) -> dict:
        """
        Checks that the given data has a counts format.

        Args:
            datum: An instance of data the should be a dict with bit strings as keys
                and counts as values.
            validate: If True the DataAction checks that the format of the datum is valid.

        Returns:
            The datum as given.

        Raises:
            DataProcessorError: if the data is not a counts dict.
        """
        if validate:
            if not isinstance(datum, dict):
                raise DataProcessorError(
                    f"Given counts datum {datum} to "
                    f"{self.__class__.__name__} is not a valid count format."
                )

            for bit_str, count in datum.items():
                if not isinstance(bit_str, str):
                    raise DataProcessorError(
                        f"Key {bit_str} is not a valid count key in{self.__class__.__name__}."
                    )

                if not isinstance(count, (int, float)):
                    raise DataProcessorError(
                        f"Count {bit_str} is not a valid count value in {self.__class__.__name__}."
                    )

        return datum

    def _process(self, datum: Dict[str, Any]) -> Tuple[float, float]:
        """
        Args:
            datum: The data dictionary,taking the data under counts and
                adding the corresponding probabilities.

        Returns:
            processed data: A dict with the populations.
        """
        shots = sum(datum.values())
        p_mean = datum.get(self._outcome, 0.0) / shots
        p_var = p_mean * (1 - p_mean) / shots

        return p_mean, p_var
