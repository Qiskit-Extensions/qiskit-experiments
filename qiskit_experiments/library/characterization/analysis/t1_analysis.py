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
T1 Analysis class.
"""
from typing import Union

import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import Options
from qiskit_experiments.curve_analysis.curve_data import CurveData


class T1Analysis(curve.DecayAnalysis):
    r"""A class to analyze T1 experiments.

    # section: see_also
        qiskit_experiments.curve_analysis.standard_analysis.decay.DecayAnalysis

    """

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.curve_drawer.set_options(
            xlabel="Delay",
            ylabel="P(1)",
            xval_unit="s",
        )
        options.result_parameters = [curve.ParameterRepr("tau", "T1", "s")]

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - absolute amp is within [0.9, 1.1]
            - base is less than 0.1
            - amp error is less than 0.1
            - tau error is less than its value
            - base error is less than 0.1
        """
        amp = fit_data.ufloat_params["amp"]
        tau = fit_data.ufloat_params["tau"]
        base = fit_data.ufloat_params["base"]

        criteria = [
            fit_data.reduced_chisq < 3,
            abs(amp.nominal_value - 1.0) < 0.1,
            abs(base.nominal_value) < 0.1,
            curve.utils.is_error_not_significant(amp, absolute=0.1),
            curve.utils.is_error_not_significant(tau),
            curve.utils.is_error_not_significant(base, absolute=0.1),
        ]

        if all(criteria):
            return "good"

        return "bad"


class T1KerneledAnalysis(curve.DecayAnalysis):
    r"""A class to analyze T1 experiments with kerneled data.

    # section: see_also
        qiskit_experiments.curve_analysis.standard_analysis.decay.DecayAnalysis

    """

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.curve_drawer.set_options(
            xlabel="Delay",
            ylabel="Normalized Projection on the Main Axis",
            xval_unit="s",
        )
        options.result_parameters = [curve.ParameterRepr("tau", "T1", "s")]
        options.normalization = True

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - absolute amp is within [0.9, 1.1]
            - base is less than 0.1
            - amp error is less than 0.1
            - tau error is less than its value
            - base error is less than 0.1
        """
        amp = fit_data.ufloat_params["amp"]
        tau = fit_data.ufloat_params["tau"]
        base = fit_data.ufloat_params["base"]

        criteria = [
            fit_data.reduced_chisq < 3,
            abs(amp.nominal_value - 1.0) < 0.1,
            abs(base.nominal_value) < 0.1,
            curve.utils.is_error_not_significant(amp, absolute=0.1),
            curve.utils.is_error_not_significant(tau),
            curve.utils.is_error_not_significant(base, absolute=0.1),
        ]

        # if amp.nominal_value > 0:
        #     criteria = [
        #         fit_data.reduced_chisq < 3,
        #         abs(amp.nominal_value - 1.0) < 0.1,
        #         abs(base.nominal_value) < 0.1,
        #         curve.utils.is_error_not_significant(amp, absolute=0.1),
        #         curve.utils.is_error_not_significant(tau),
        #         curve.utils.is_error_not_significant(base, absolute=0.1),
        #     ]
        # else:
        #     # In SVD decomposition, the main vector is determined up to its sign.Therefore, two graphs
        #     # can fit our data. The fit could either be `a*exp(-t/tau)+b` or `1-a*exp(-t/tau)+b` where
        #     # a=1 and b=0. for the second one, we can alternatively fit it to `a*exp(-t/tau)+b` with
        #     # a=-1 and b=1.
        #     criteria = [
        #         fit_data.reduced_chisq < 3,
        #         abs(amp.nominal_value + 1.0) < 0.1,
        #         abs(base.nominal_value - 1) < 0.1,
        #         curve.utils.is_error_not_significant(amp, absolute=0.1),
        #         curve.utils.is_error_not_significant(tau),
        #         curve.utils.is_error_not_significant(base, absolute=0.1),
        #     ]

        if all(criteria):
            return "good"

        return "bad"

    def _format_data(
        self,
        curve_data: curve.CurveData,
    ) -> curve.CurveData:
        """Postprocessing for the processed dataset.

        Args:
            curve_data: Processed dataset created from experiment results.

        Returns:
            Formatted data.
        """
        # check if the SVD decomposition categorized 0 as 1
        if curve_data.y[-1] == 1:
            new_y_data = np.zeros(curve_data.y.shape)
            for idx, y_data in enumerate(curve_data.y):
                new_y_data[idx] = 1 - (y_data - curve_data.y[1])

            new_curve_data = CurveData(
                x=curve_data.x,
                y=new_y_data,
                y_err=curve_data.y_err,
                shots=curve_data.shots,
                data_allocation=curve_data.data_allocation,
                labels=curve_data.labels,
            )

            return super()._format_data(new_curve_data)
        return super()._format_data(curve_data)
