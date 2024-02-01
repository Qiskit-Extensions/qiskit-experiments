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

"""Restless mixin class."""

import logging
from typing import Callable, Sequence, Optional
from qiskit.qobj.utils import MeasLevel, MeasReturnType

from qiskit.providers import Backend
from qiskit_experiments.framework import Options
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.nodes import ProjectorType
from qiskit_experiments.data_processing import nodes
from qiskit_experiments.data_processing.processor_library import get_kerneled_processor
from qiskit_experiments.framework.base_analysis import BaseAnalysis

LOG = logging.getLogger(__name__)


class RestlessMixin:
    """A mixin to facilitate restless experiments.

    This class defines the following methods:

    - :meth:`~.RestlessMixin.enable_restless`
    - :meth:`~.RestlessMixin._enable_restless`
    - :meth:`~.RestlessMixin._get_restless_processor`
    - :meth:`~.RestlessMixin._t1_check`
    - :meth:`~.RestlessMixin._finalize`

    A restless enabled experiment is an experiment that can be run in a restless
    measurement setting. In restless measurements, the qubit is not reset after
    each measurement. Instead, the outcome of the previous quantum non-demolition
    measurement is the initial state for the current circuit. Restless measurements
    therefore require special data processing which is provided by sub-classes of
    the :class:`.RestlessNode`. Restless experiments are a fast alternative for
    several calibration and characterization tasks, for details see
    https://arxiv.org/pdf/2202.06981.pdf.

    This class makes it possible for users to enter a restless run mode without having
    to manually set all the required run options and the data processor. The required options
    are ``rep_delay``, ``init_qubits``, ``memory``, and ``meas_level``. Furthermore,
    subclasses can override the :meth:`_get_restless_processor` method if they require more
    complex restless data processing such as two-qubit calibrations. In addition, this
    class makes it easy to determine if restless measurements are supported for a given
    experiment.

    User Manual
        :doc:`/manuals/measurement/restless_measurements`

    """

    analysis: BaseAnalysis
    _default_run_options: Options()
    run_options: Options()
    set_run_options: Callable
    _backend: Backend
    _physical_qubits: Sequence[int]
    _num_qubits: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._enable_restless = False
        self.rep_delay = None
        self.override_processor_by_restless = None
        self.suppress_t1_error = None

    def enable_restless(
        self,
        rep_delay: Optional[float] = None,
        override_processor_by_restless: bool = True,
        suppress_t1_error: bool = False,
    ):
        """
        Set a flag to enable restless configuration when running the experiment. If the flag is set to
        `True`, the program will run `_enable_restless` method in the `_finilized` step.

        Args:
            rep_delay: The repetition delay. This is the delay between a measurement
                and the subsequent quantum circuit. Since the backends have
                dynamic repetition rates, the repetition delay can be set to a small
                value which is required for restless experiments. Typical values are
                1 us or less.
            override_processor_by_restless: If False, a data processor that is specified in the
                analysis options of the experiment is not overridden by the restless data
                processor. The default is True.
            suppress_t1_error: If True, the default is False, then no error will be raised when
                ``rep_delay`` is larger than the T1 times of the qubits. Instead, a warning will
                be logged as restless measurements may have a large amount of noise.

        """

        self.set_run_options(restless=True)
        if rep_delay:
            self.rep_delay = rep_delay
        if override_processor_by_restless:
            self.override_processor_by_restless = override_processor_by_restless
        if suppress_t1_error:
            self.suppress_t1_error = suppress_t1_error

        # Calling `_enable_restless()` so the run option will be visible to the user.
        self._enable_restless()
        # Setting override_processor_by_restless to false because
        # the data processor was already configured.
        self.override_processor_by_restless = False

    def _enable_restless(self):
        """Enables a restless experiment by setting the restless run options and the
        restless data processor.

        Raises:
            DataProcessorError: If the attribute rep_delay_range is not defined for the backend.
            DataProcessorError: If a data processor has already been set but
                override_processor_by_restless is True.
            DataProcessorError: If the experiment analysis does not have the data_processor
                option.
            DataProcessorError: If the rep_delay is equal to or greater than the
                T1 time of one of the physical qubits in the experiment and the flag
                ``ignore_t1_check`` is False.
        """
        LOG.debug("Enabling restless configuration. This will override current configuration.")
        try:
            if not self.rep_delay:
                # BackendV1 only; BackendV2 does not support this
                self.rep_delay = self._backend.configuration().rep_delay_range[0]
        except AttributeError as error:
            raise DataProcessorError(
                "The restless experiment can not be enabled because "
                "the attribute rep_delay_range is not defined for this backend "
                "and a minimum rep_delay can not be set."
            ) from error

        # Check the rep_delay compared to the T1 time.
        if not self._t1_check():
            msg = (
                f"The specified repetition delay {self.rep_delay} is equal to or greater "
                f"than the T1 time of one of the physical qubits"
                f"{self._physical_qubits} in the experiment. Consider choosing "
                f"a smaller repetition delay for the restless experiment."
            )

            if self.suppress_t1_error:
                LOG.warning(msg)
            else:
                raise DataProcessorError(msg)

        # The excited state promotion readout analysis option is set to
        # False because it is not compatible with a restless experiment.
        meas_level = self._default_run_options().get("meas_level", MeasLevel.CLASSIFIED)
        meas_return = self._default_run_options().get("meas_return", MeasReturnType.SINGLE)
        if not self.analysis.options.get("data_processor", None):
            self.set_run_options(
                rep_delay=self.rep_delay,
                init_qubits=False,
                memory=True,
                meas_level=meas_level,
                meas_return=meas_return,
                use_measure_esp=False,
            )
            if hasattr(self.analysis.options, "data_processor"):
                self.analysis.set_options(
                    data_processor=self._get_restless_processor(meas_level=meas_level)
                )
            else:
                raise DataProcessorError(
                    "The restless data processor can not be set since the experiment analysis"
                    "does not have the data_processor option."
                )
        else:
            if not self.override_processor_by_restless:
                self.set_run_options(
                    rep_delay=self.rep_delay,
                    init_qubits=False,
                    memory=True,
                    meas_level=meas_level,
                    meas_return=meas_return,
                    use_measure_esp=False,
                )
            else:
                raise DataProcessorError(
                    "Cannot enable restless. Data processor has already been set and "
                    "override_processor_by_restless is True."
                )

    def _get_restless_processor(self, meas_level: int = MeasLevel.CLASSIFIED) -> DataProcessor:
        """Returns the restless experiments data processor.

        Notes:
            Sub-classes can override this method if they need more complex data processing.
        """
        outcome = self.analysis.options.get("outcome", "1" * self._num_qubits)
        meas_return = self.analysis.options.get("meas_return", MeasReturnType.SINGLE)
        normalize = self.analysis.options.get("normalization", True)
        dimensionality_reduction = self.analysis.options.get(
            "dimensionality_reduction", ProjectorType.SVD
        )

        if meas_level == MeasLevel.KERNELED:
            return get_kerneled_processor(
                dimensionality_reduction, meas_return, normalize, [nodes.RestlessToIQ()]
            )

        else:
            return DataProcessor(
                "memory",
                [
                    nodes.RestlessToCounts(self._num_qubits),
                    nodes.Probability(outcome),
                ],
            )

    def _t1_check(self) -> bool:
        """Check that repetition delay < T1 of the physical qubits in the experiment.

        Returns:
            True if the repetition delay is smaller than the qubit T1 times.

        Raises:
            DataProcessorError: If the T1 values are not defined for the qubits of
                the used backend.
        """

        try:
            t1_values = [
                self._backend_data.qubit_t1(physical_qubit)
                for physical_qubit in self._physical_qubits
            ]

            if all(self.rep_delay / t1_value < 1.0 for t1_value in t1_values):
                return True
        except AttributeError as error:
            raise DataProcessorError(
                "The restless experiment can not be enabled since "
                "T1 values are not defined for the qubits of the used backend."
            ) from error

        return False

    def _finalize(self):
        # Calling again to self._enable_restless() to override experiment option for restless experiment.
        if self.run_options.get("restless", None):
            self._enable_restless()
        super()._finalize()
