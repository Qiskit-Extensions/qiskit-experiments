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
"""Decay analysis class."""

from typing import List, Union, Tuple

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import ExperimentData, AnalysisResultData, Options, FitVal
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.processor_library import get_processor


class DecayAnalysis(curve.CurveAnalysis):
    r"""A class to analyze general exponential decay curve.

    # section: fit_model

    The fit is based on the following decay function.

    .. math::
        F(x) = {\rm amp} \cdot e^{-x/\tau} + {\rm base}

    # section: fit_parameters

        defpar \rm amp:
           desc: Height of the decay curve.
           init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.guess.min_height`.
           bounds: None

        defpar \rm base:
           desc: Base line of the decay curve.
           init_guess: Determined by the difference of minimum and maximum points.
           bounds: None

        defpar \tau:
           desc: This is the fit parameter of main interest.
           init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.guess.exp_decay`.
           bounds: None

    """

    def __init__(self):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x/tau) + base",
                    name="exp_decay",
                )
            ]
        )

    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            raw_data_mode (bool): If the option is set to ``True`` the curve fitting is skipped and
                returns raw data point at each scan point. This option is sometime useful for experiments
                that aims at measuring fast dynamics or scan decay values with various configurations.
        """
        options = super()._default_options()
        options.raw_data_mode = False

        return options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.p0.set_if_empty(base=curve.guess.min_height(curve_data.y)[0])

        alpha = curve.guess.exp_decay(curve_data.x, curve_data.y)

        if alpha != 0.0:
            user_opt.p0.set_if_empty(
                tau=-1 / alpha,
                amp=curve.guess.max_height(curve_data.y)[0] - user_opt.p0["base"],
            )
        else:
            # Likely there is no slope. Cannot fit constant line with this model.
            # Set some large enough number against to the scan range.
            user_opt.p0.set_if_empty(
                tau=100 * np.max(curve_data.x),
                amp=curve.guess.max_height(curve_data.y)[0] - user_opt.p0["base"],
            )
        return user_opt

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - tau error is less than its value
        """
        tau = fit_data.ufloat_params["tau"]

        criteria = [
            fit_data.reduced_chisq < 3,
            curve.utils.is_error_not_significant(tau),
        ]

        if all(criteria):
            return "good"

        return "bad"

    def _run_analysis(
        self, experiment_data: ExperimentData, **options
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:
        """Run analysis on circuit data.

        Args:
            experiment_data: the experiment data to analyze.
            options: kwarg options for analysis function.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where ``analysis_results``
                   is a list of :class:`AnalysisResultData` objects, and ``figures``
                   is a list of any figures for the experiment.
        """
        # TODO this will be cleaned up with analysis configuration refactoring
        analysis_options = self._default_options().__dict__
        analysis_options.update(options)

        if analysis_options["raw_data_mode"]:
            # TODO we need separation of data processing. Below is boilerplate code just copied from base class.
            self._arg_parse(**analysis_options)

            data_processor = self._get_option("data_processor")

            if not data_processor:
                run_options = self._run_options() or dict()

                try:
                    meas_level = run_options["meas_level"]
                except KeyError as ex:
                    raise DataProcessorError(
                        f"Cannot process data without knowing the measurement level: {str(ex)}."
                    ) from ex

                meas_return = run_options.get("meas_return", None)
                normalization = self._get_option("normalization")

                data_processor = get_processor(meas_level, meas_return, normalization)

            if isinstance(data_processor, DataProcessor) and not data_processor.is_trained:
                # Qiskit DataProcessor instance. May need calibration.
                data_processor.train(data=experiment_data.data())

            self._extract_curves(experiment_data=experiment_data, data_processor=data_processor)

            raw_data = self._data()

            analysis_results = []
            for xval, yval, yerr in zip(raw_data.x, raw_data.y, raw_data.y_err):
                analysis_results.append(
                    AnalysisResultData(
                        name=f"{self.__class__.__name__}_single_point",
                        value=FitVal(yval, yerr),
                        extra={
                            "xval": xval,
                        },
                    )
                )

            return analysis_results, []
        else:
            return super()._run_analysis(experiment_data, **analysis_options)
