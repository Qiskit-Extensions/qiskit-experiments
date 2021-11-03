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

# pylint: disable=arguments-differ

"""Different data analysis steps."""

from numbers import Number
from typing import List, Tuple, Union, Sequence, Generator, Iterator, Optional

import numpy as np

from qiskit_experiments.data_processing.data_action import DataAction, TrainableDataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from .data_processor import execute_pipeline


class AverageData(DataAction):
    """A node to average data representable as numpy arrays."""

    def __init__(self, axis: int, validate: bool = True):
        """Initialize a data averaging node.

        Args:
            axis: The axis along which to average.
            validate: If set to False the DataAction will not validate its input.

        Note:
            Axis depends on data type. ``axis = 0`` indicates averaging over
            different circuits in the experiment data. Level2 data has only this axis.
            As a reminder the table below shows that various data levels in Qiskit
            and their dimension. The AverageData node will simply average the
            given array over the specified axis.

                ============  =============  =====
                `meas_level`  `meas_return`  shape
                ============  =============  =====
                0             `single`       np.ndarray[shots, memory_slots, memory_slot_size]
                0             `avg`          np.ndarray[memory_slots, memory_slot_size]
                1             `single`       np.ndarray[shots, memory_slots]
                1             `avg`          np.ndarray[memory_slots]
                2             `memory=True`  list
                ============  =============  =====

        """
        super().__init__(validate)
        self._axis = axis

    def _format_data(self, gen_datum: Iterator) -> Generator:
        """Format and validate.

        Args:
             gen_datum: A pipeline.

        Yields:
            A formatted value array. Error is discarded.

        Raises:
            DataProcessorError: When non-existing data axis is specified.
        """
        for value_array, error_array in gen_datum:
            value_array = np.asarray(value_array, dtype=float)
            error_array = np.asarray(error_array, dtype=float)

            if self._validate:
                # shape is reduced because this is a single entry
                if len(value_array.shape) <= self._axis - 1:
                    raise DataProcessorError(
                        f"Cannot average the {len(value_array.shape)} dimensional "
                        f"array along axis {self._axis}."
                    )

            yield value_array, error_array

    def _process(self, gen_datum: Iterator) -> Generator:
        """Average the data.

         Args:
             gen_datum: A pipeline.

        Yields:
             Two arrays with one less dimension than the given datum and error. The error
             is the standard error of the mean, i.e. the standard deviation of the datum
             divided by :math:`sqrt{N}` where :math:`N` is the number of data points.
        """
        if self._axis == 0:
            # average over different circuits. execute pipeline.
            full_val_arr, _ = execute_pipeline(gen_datum)
            n_circs = full_val_arr.shape[0]

            # take average over full matrix
            avg_mat = np.average(full_val_arr, axis=self._axis)
            std_mat = np.std(full_val_arr, axis=self._axis) / np.sqrt(n_circs)

            yield np.asarray(avg_mat, dtype=float), np.asarray(std_mat, dtype=float)
        else:
            # keep pipeline, e.g. averaging over shots of single circuit
            for value_array, _ in gen_datum:
                axis = self._axis - 1
                n_elements = value_array.shape[axis]

                avg_mat = np.average(value_array, axis=axis)
                std_mat = np.std(value_array, axis=axis) / np.sqrt(n_elements)

                yield avg_mat, std_mat


class MinMaxNormalize(DataAction):
    """Normalizes the data."""

    def _format_data(self, gen_datum: Iterator) -> Tuple[np.ndarray, np.ndarray]:
        """Format and validate.

        Args:
             gen_datum: A pipeline.

        Returns:
            A tuple of formatted values and error arrays.
        """
        full_val_arr, full_err_arr = execute_pipeline(gen_datum)
        return np.asarray(full_val_arr, dtype=float), np.asarray(full_err_arr, dtype=float)

    def _process(self, full_arrays_tup: Tuple[np.ndarray, np.ndarray]) -> Generator:
        """Normalzie data. This node execute pipeline and generate full data array.

         Args:
             full_arrays_tup: Values and errors from executed pipeline.

        Yields:
             Values normalized to the interval [0, 1].
        """
        full_val_arr, full_err_arr = full_arrays_tup

        min_y, max_y = np.min(full_val_arr), np.max(full_val_arr)
        scale = float(max_y) - float(min_y)

        for out_value, out_error in zip(full_val_arr, full_err_arr):
            yield (out_value - min_y) / scale, out_error / scale


class SVD(TrainableDataAction):
    """Singular Value Decomposition of averaged IQ data."""

    def __init__(self, validate: bool = True):
        """Create new action.

        Args:
            validate: If set to False the DataAction will not validate its input.
        """
        super().__init__(validate=validate)
        self._main_axes = None
        self._means = None
        self._scales = None
        self._n_shots = 0
        self._n_slots = 0
        self._n_iq = 0

    def _format_data(self, gen_datum: Iterator) -> Generator:
        """Format and validate.

        Args:
             gen_datum: A pipeline.

        Yields:
            A tuple of formatted data and error.

        Raises:
            DataProcessorError: If the datum does not have the correct format.
        """
        for value_array, error_array in gen_datum:
            value_array = np.asarray(value_array, dtype=float)
            error_array = np.asarray(error_array, dtype=float)
            self._n_shots = 0
            self._n_slots = 0
            self._n_iq = 0

            # identify shape
            try:
                # level1 single mode
                self._n_shots, self._n_slots, self._n_iq = value_array.shape
            except ValueError:
                try:
                    # level1 average mode
                    self._n_slots, self._n_iq = value_array.shape
                except ValueError as ex:
                    raise DataProcessorError(
                        f"Data given to {self.__class__.__name__} is not likely level1 data."
                    ) from ex

            if self._validate:
                if self._n_iq != 2:
                    raise DataProcessorError(
                        f"IQ data given to {self.__class__.__name__} must be a 2D array. "
                        f"Instead, a {self._n_iq}D array was given."
                    )
                if value_array.shape != error_array.shape:
                    raise DataProcessorError(
                        f"IQ data error given to {self.__class__.__name__} is invalid data shape."
                    )

            yield value_array, error_array

    def _process(self, gen_datum: Iterator) -> Generator:
        """Compute singular values.

        Args:
            gen_datum: A generator of unprocessed data. Each entry is a tuple of data and error.

        Yields:
            A tuple of processed data and error.

        Raises:
            DataProcessorError: If the SVD has not been previously trained on data.
        """
        if not self.is_trained:
            raise DataProcessorError("SVD must be trained on data before it can be used.")

        for value_array, error_array in gen_datum:
            if self._n_shots == 0:
                # level1 single mode, IQ axis is projected
                singular_vals = np.zeros(self._n_slots, dtype=float)
                error_vals = np.zeros(self._n_slots, dtype=float)
            else:
                # level1 average mode, IQ axis is projected
                singular_vals = np.zeros((self._n_shots, self._n_slots), dtype=float)
                error_vals = np.zeros((self._n_shots, self._n_slots), dtype=float)

            # process each averaged IQ point with its own axis.
            for idx in range(self._n_slots):
                scale = self.scales[idx]
                centered = np.array(
                    [
                        value_array[..., idx, iq] - self.means(qubit=idx, iq_index=iq)
                        for iq in [0, 1]
                    ]
                )
                angle = np.arctan(self._main_axes[idx][1] / self._main_axes[idx][0])

                singular_vals[..., idx] = (self._main_axes[idx] @ centered) / scale
                error_vals[..., idx] = (
                    np.sqrt(
                        (error_array[..., idx, 0] * np.cos(angle)) ** 2
                        + (error_array[..., idx, 1] * np.sin(angle)) ** 2
                    )
                    / scale
                )

            yield singular_vals, error_vals

    @property
    def axis(self) -> List[np.array]:
        """Return the axis of the trained SVD"""
        return self._main_axes

    def means(self, qubit: int, iq_index: int) -> float:
        """Return the mean by which to correct the IQ data.

        Before training the SVD the mean of the training data is subtracted from the
        training data to avoid large offsets in the data. These means can be retrieved
        with this function.

        Args:
            qubit: Index of the qubit.
            iq_index: Index of either the in-phase (i.e. 0) or the quadrature (i.e. 1).

        Returns:
            The mean that was determined during training for the given qubit and IQ index.
        """
        return self._means[qubit][iq_index]

    @property
    def scales(self) -> List[float]:
        """Return the scaling of the SVD."""
        return self._scales

    @property
    def is_trained(self) -> bool:
        """Return True is the SVD has been trained.

        Returns:
            True if the SVD has been trained.
        """
        return self._main_axes is not None

    def train(self, full_val_arr: np.ndarray, full_err_arr: Optional[np.ndarray] = None):
        """Train the SVD on the given data.

        Each element of the given data will be converted to a 2D array of dimension
        n_qubits x 2. The number of qubits is inferred from the shape of the data.
        For each qubit the data is collected into an array of shape 2 x n_data_points.
        The mean of the in-phase a quadratures is subtracted before passing the data
        to numpy's svd function. The dominant axis and the scale is saved for each
        qubit so that future data points can be projected onto the axis.

        Args:
            full_val_arr: A list of values. Each datum will be converted to a 2D array.
            full_err_arr: A list of errors. Each datm will be converted to a 2D array.
        """
        if full_val_arr is None:
            return

        # Format
        full_val_arr = np.asarray(full_val_arr, dtype=float)

        if full_err_arr is None:
            full_err_arr = np.full_like(full_val_arr, np.nan, dtype=float)

        # TODO should consider error
        _ = np.asarray(full_err_arr, dtype=float)

        self._main_axes = []
        self._scales = []
        self._means = []
        n_slots = full_val_arr.shape[1]
        for slot_idx in range(n_slots):
            datums = np.vstack([datum[slot_idx] for datum in full_val_arr]).T

            # Calculate the mean of the data to recenter it in the IQ plane.
            mean_i = np.average(datums[0, :])
            mean_q = np.average(datums[1, :])

            self._means.append((mean_i, mean_q))

            datums[0, :] = datums[0, :] - mean_i
            datums[1, :] = datums[1, :] - mean_q

            mat_u, mat_s, _ = np.linalg.svd(datums)

            self._main_axes.append(mat_u[:, 0])
            self._scales.append(mat_s[0])


class IQPart(DataAction):
    """Abstract class for IQ data post-processing."""

    def __init__(self, scale: float = 1.0, validate: bool = True):
        """Create new action.

        Args:
            scale: Float with which to multiply the IQ data. Defaults to 1.0.
            validate: If set to False the DataAction will not validate its input.
        """
        self.scale = scale
        super().__init__(validate)

    def _process(self, gen_datum: Iterator) -> Generator:
        """Defines how the IQ point is processed.

        The dimension of the input datum corresponds to different types of data:
        - 2D represents average IQ Data.
        - 3D represents either a single-shot datum or all data of averaged data.
        - 4D represents all data of single-shot data.

        Args:
            gen_datum: A generator of unprocessed data. Each entry is a tuple of data and error.

        Yields:
            A tuple of processed data and error.
        """
        raise NotImplementedError

    def _format_data(self, gen_datum: Iterator) -> Generator:
        """Validate and format the input.

        Check that the given data and error have the correct structure.

        Args:
            gen_datum: A generator of unformatted data. Each entry is a tuple of data and error.

        Yields:
            A tuple of formatted data and error.

        Raises:
            DataProcessorError: If the datum does not have the correct format.
        """
        for value_array, error_array in gen_datum:
            value_array = np.asarray(value_array, dtype=float)
            error_array = np.asarray(error_array, dtype=float)

            if self._validate:
                if len(value_array.shape) not in {1, 2, 3}:
                    raise DataProcessorError(
                        f"IQ data given to {self.__class__.__name__} must be an N dimensional"
                        "array with N in (1, 2, 3). "
                        f"Instead, a {len(value_array.shape)}D array was given."
                    )

                if len(error_array.shape) not in {1, 2, 3}:
                    raise DataProcessorError(
                        f"IQ data error given to {self.__class__.__name__} must be an N dimensional"
                        "array with N in (1, 2, 3). "
                        f"Instead, a {len(error_array.shape)}D array was given."
                    )

                if len(error_array.shape) != len(value_array.shape):
                    raise DataProcessorError(
                        "Datum and error do not have the same shape: "
                        f"{len(value_array.shape)} != {len(error_array.shape)}."
                    )
            yield value_array, error_array

    def __repr__(self):
        """String representation of the node."""
        return f"{self.__class__.__name__}(validate: {self._validate}, scale: {self.scale})"


class ToReal(IQPart):
    """IQ data post-processing. Isolate the real part of single-shot IQ data."""

    def _process(self, gen_datum: Iterator) -> Generator:
        """Take the real part of the IQ data.

        Args:
            gen_datum: A generator of unprocessed data. Each entry is a tuple of data and error.

        Yields:
            A tuple of processed data and error. A N-1 dimensional array,
            each entry is the real part of the given IQ data and error.
        """
        for value_array, error_array in gen_datum:
            yield value_array[..., 0] * self.scale, error_array[..., 0] * self.scale


class ToImag(IQPart):
    """IQ data post-processing. Isolate the imaginary part of single-shot IQ data."""

    def _process(self, gen_datum: Iterator) -> Generator:
        """Take the imaginary part of the IQ data.

        Args:
            gen_datum: A generator of unprocessed data. Each entry is a tuple of data and error.

        Yields:
            A tuple of processed data and error. A N-1 dimensional array,
            each entry is the imaginary part of the given IQ data and error.
        """
        for value_array, error_array in gen_datum:
            yield value_array[..., 1] * self.scale, error_array[..., 1] * self.scale


class Probability(DataAction):
    r"""Compute the mean probability of a single measurement outcome from counts.

    This node returns the mean and standard deviation of a single measurement
    outcome probability :math:`p` estimated from the observed counts. The mean and
    variance are computed from the posterior Beta distribution
    :math:`B(\alpha_0^\prime,\alpha_1^\prime)` estimated from a Bayesian update
    of a prior Beta distribution :math:`B(\alpha_0, \alpha_1)` given the observed
    counts.

    The mean and variance of the Beta distribution :math:`B(\alpha_0, \alpha_1)` are:

    .. math::

        \text{E}[p] = \frac{\alpha_0}{\alpha_0 + \alpha_1}, \quad
        \text{Var}[p] = \frac{\text{E}[p] (1 - \text{E}[p])}{\alpha_0 + \alpha_1 + 1}

    Given a prior Beta distribution :math:`B(\alpha_0, \alpha_1)`, the posterior
    distribution for the observation of :math:`F` counts of a given
    outcome out of :math:`N` total shots is a
    :math:`B(\alpha_0^\prime,\alpha_1^\prime):math:` with

    .. math::
        \alpha_0^\prime = \alpha_0 + F, \quad
        \alpha_1^\prime = \alpha_1 + N - F.

    .. note::

        The default value for the prior distribution is *Jeffery's Prior*
        :math:`\alpha_0 = \alpha_1 = 0.5` which represents ignorance about the true
        probability value. Note that for this prior the mean probability estimate
        from a finite number of counts can never be exactly 0 or 1. The estimated
        mean and variance are given by

        .. math::

            \text{E}[p] = \frac{F + 0.5}{N + 1}, \quad
            \text{Var}[p] = \frac{\text{E}[p] (1 - \text{E}[p])}{N + 2}
    """

    def __init__(
        self,
        outcome: str,
        alpha_prior: Union[float, Sequence[float]] = 0.5,
        validate: bool = True,
    ):
        """Initialize a counts to probability data conversion.

        Args:
            outcome: The bitstring for which to return the probability and variance.
            alpha_prior: A prior Beta distribution parameter ``[`alpha0, alpha1]``.
                         If specified as float this will use the same value for
                         ``alpha0`` and``alpha1`` (Default: 0.5).
            validate: If set to False the DataAction will not validate its input.

        Raises:
            DataProcessorError: When the dimension of the prior and expected parameter vector
                do not match.
        """
        self._outcome = outcome
        if isinstance(alpha_prior, Number):
            self._alpha_prior = [alpha_prior, alpha_prior]
        else:
            if validate and len(alpha_prior) != 2:
                raise DataProcessorError(
                    "Prior for probability node must be a float or pair of floats."
                )
            self._alpha_prior = list(alpha_prior)
        super().__init__(validate)

    def _format_data(self, gen_datum: Iterator) -> Generator:
        """
        Checks that the given data has a counts format.

        Args:
            gen_datum: A generator of unformatted data. Each entry is a tuple of data and error.

        Yields:
            The datum typecasted to dictionary.

        Raises:
            DataProcessorError: if the data is not a counts dict or a list of counts dicts.
        """
        for value_array, _ in gen_datum:
            # Discard previous data. Probability is determined by sampling error.
            # Any IQ distribution variance will be ignored at this stage.
            count_dict = value_array[0]

            if self._validate:
                if not isinstance(count_dict, dict):
                    raise DataProcessorError(
                        f"Given counts datum {count_dict} to "
                        f"{self.__class__.__name__} is not a valid count format."
                    )
                for bit_str, count in count_dict.items():
                    if not isinstance(bit_str, str):
                        raise DataProcessorError(
                            f"Key {bit_str} is not a valid count key in {self.__class__.__name__}."
                        )
                    if not isinstance(count, (int, float, np.integer)):
                        raise DataProcessorError(
                            f"Count {bit_str} is not a valid count value in {self.__class__.__name__}."
                        )

            yield count_dict

    def _process(self, gen_datum: Iterator) -> Generator:
        """Compute probability and sampling error.

        Args:
            gen_datum: A generator of unprocessed data. Each entry is a count dictionary.

        Yields:
            A tuple of processed data and error.
        """
        for count_dict in gen_datum:

            shots = sum(count_dict.values())
            freq = count_dict.get(self._outcome, 0)
            alpha_posterior = [freq + self._alpha_prior[0], shots - freq + self._alpha_prior[1]]
            alpha_sum = sum(alpha_posterior)
            p_mean = alpha_posterior[0] / alpha_sum
            p_var = p_mean * (1 - p_mean) / (alpha_sum + 1)

            yield np.asarray([p_mean], dtype=float), np.asarray([np.sqrt(p_var)], dtype=float)


class BasisExpectationValue(DataAction):
    """Compute expectation value of measured basis from probability.

    Note:
        The sign becomes P(0) -> 1, P(1) -> -1.
    """

    def _format_data(self, gen_datum: Iterator) -> Generator:
        """Validate and format the input.

        Check if given value is likely probability.

        Args:
            gen_datum: A generator of unformatted data. Each entry is a tuple of data and error.

        Yields:
            A tuple of formatted data and error.

        Raises:
            DataProcessorError: When input value is not in [0, 1]
        """
        for value_array, error_array in gen_datum:
            value = float(value_array)
            error = float(error_array)
            if self._validate:
                if not 0 < value < 1:
                    raise DataProcessorError(
                        f"Input data for node {self.__class__.__name__} is not likely probability."
                    )
            yield value, error

    def _process(self, gen_datum: Iterator) -> Generator:
        """Compute eigenvalue.

        Args:
            gen_datum: A generator of unprocessed data. Each entry is a tuple of data and error.

        Yields:
            A tuple of processed data and error.
        """
        for value, stdev in gen_datum:
            yield np.asarray(2 * (0.5 - value), dtype=float), np.asarray(2 * stdev, dtype=float)
