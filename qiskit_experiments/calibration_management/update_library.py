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

"""A library of experiment calibrations."""

from abc import ABC
from datetime import datetime, timezone
from typing import Optional, Union

from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.calibration_management.base_calibrations import BaseCalibrations
from qiskit_experiments.calibration_management.parameter_value import ParameterValue
from qiskit_experiments.calibration_management.calibration_key_types import ParameterValueType
from qiskit_experiments.exceptions import CalibrationError


class BaseUpdater(ABC):
    """A base class to update calibrations."""

    __fit_parameter__ = None

    def __init__(self):
        """Updaters are not meant to be instantiated.

        Instead of instantiating updaters use them by calling the :meth:`update` class method.
        For example, the :class:`.Frequency` updater is called in the following way

         .. code-block:: python

            Frequency.update(calibrations, spectroscopy_data)

        Here, calibrations is an instance of :class:`.Calibrations` and spectroscopy_data
        is the result of a :class:`.QubitSpectroscopy` experiment.
        """
        raise CalibrationError(
            "Calibration updaters are not meant to be instantiated. The intended usage"
            "is Updater.update(calibrations, exp_data, ...)."
        )

    @staticmethod
    def _time_stamp(exp_data: ExperimentData) -> datetime:
        """Helper method to extract the datetime."""
        all_times = exp_data.completion_times.values()
        if all_times:
            return max(all_times).astimezone()
        return datetime.now(timezone.utc).astimezone()

    @classmethod
    def add_parameter_value(
        cls,
        cal: BaseCalibrations,
        exp_data: ExperimentData,
        value: ParameterValueType,
        param: Union[Parameter, str],
        schedule: Union[ScheduleBlock, str] = None,
        group: str = "default",
    ):
        """Update the calibrations with the given value.

        Args:
            cal: The Calibrations instance to update.
            exp_data: The ExperimentData instance that contains the result and the experiment data.
            value: The value extracted by the subclasses in the :meth:`update` method.
            param: The name of the parameter, or the parameter instance, which will receive an
                updated value.
            schedule: The ScheduleBlock instance or the name of the instance to which the parameter
                is attached.
            group: The calibrations group to update.
        """

        qubits = exp_data.metadata["physical_qubits"]

        param_value = ParameterValue(
            value=value,
            date_time=cls._time_stamp(exp_data),
            group=group,
            exp_id=exp_data.experiment_id,
        )

        cal.add_parameter_value(param_value, param, qubits, schedule)

    @classmethod
    def update(
        cls,
        calibrations: BaseCalibrations,
        exp_data: ExperimentData,
        parameter: str,
        schedule: Optional[Union[ScheduleBlock, str]],
        result_index: Optional[int] = -1,
        group: str = "default",
        fit_parameter: Optional[str] = None,
    ):
        """Update the calibrations based on the data.

        Args:
            calibrations: The calibrations to update.
            exp_data: The experiment data from which to update.
            parameter: The name of the parameter in the calibrations to update.
            schedule: The ScheduleBlock instance or the name of the instance to which the parameter
                is attached.
            result_index: The result index to use which defaults to -1.
            group: The calibrations group to update. Defaults to "default."
            fit_parameter: The name of the fit parameter in the analysis result. This will default
                to the class variable :code:`__fit_parameter__` if not given.

        Raises:
            CalibrationError: If the analysis result does not contain a frequency variable.
        """
        fit_parameter = fit_parameter or cls.__fit_parameter__
        value = BaseUpdater.get_value(exp_data, fit_parameter, result_index)

        cls.add_parameter_value(
            calibrations, exp_data, value, parameter, schedule=schedule, group=group
        )

    @staticmethod
    def get_value(exp_data: ExperimentData, param_name: str, index: Optional[int] = -1) -> float:
        """A helper method to extract values from experiment data instances."""
        # Because this is called within analysis callbacks the block=False kwarg
        # must be passed to analysis results so we don't block indefinitely
        candidates = exp_data.analysis_results(param_name, block=False)
        if isinstance(candidates, list):
            return candidates[index].value.nominal_value
        else:
            return candidates.value.nominal_value


class Frequency(BaseUpdater):
    """Update frequencies."""

    __fit_parameter__ = "f01"

    # pylint: disable=arguments-differ,unused-argument
    @classmethod
    def update(
        cls,
        calibrations: BaseCalibrations,
        exp_data: ExperimentData,
        result_index: Optional[int] = None,
        parameter: str = "drive_freq",
        group: str = "default",
        fit_parameter: Optional[str] = None,
        **options,
    ):
        """Update a qubit frequency from, e.g., QubitSpectroscopy

        The value of the amplitude must be derived from the fit so the base method cannot be used.

        Args:
            calibrations: The calibrations to update.
            exp_data: The experiment data from which to update.
            result_index: The result index to use which defaults to -1.
            parameter: The name of the parameter to update. If None is given this will default
                to `drive_freq`.
            group: The calibrations group to update. Defaults to "default."
            options: Trailing options.
            fit_parameter: The name of the fit parameter in the analysis result. This will default
                to the class variable :code:`__fit_parameter__` if not given.

        """
        super().update(
            calibrations=calibrations,
            exp_data=exp_data,
            parameter=parameter,
            schedule=None,
            result_index=result_index,
            group=group,
            fit_parameter=fit_parameter,
        )
