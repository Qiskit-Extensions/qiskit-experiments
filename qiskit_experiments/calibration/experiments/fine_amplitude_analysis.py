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

"""Fine amplitude calibration analysis."""

from typing import Any, Dict, List, Union
import numpy as np

from qiskit_experiments.analysis import (
    CurveAnalysis,
    CurveAnalysisResult,
    SeriesDef,
    fit_function,
    get_opt_value,
    get_opt_error,
)


class FineAmplitudeAnalysis(CurveAnalysis):
    r"""Fine amplitude analysis class based on a fit to a cosine function.

    Analyse a fine amplitude calibration experiment by fitting the data to a cosine function.
    The user must also specify the intended rotation angle per gate, here labeled,
    :math:`{\rm apg}` (TODO for now this is hard coded to pi). The parameter of interest in the
    fit is the deviation from the intended rotation angle per gate labeled :math:`{\rm d}\theta`.
    The fit function is

    .. math::
        y = \frac{\rm amp}{2} \cos\left( x [{\rm d}\theta + {\rm apg} ] + {\rm phase}\right) + baseline

    Fit Parameters
        - :math:`amp`: Amplitude of the oscillation.
        - :math:`baseline`: Base line.
        - :math:`{\rm d}\theta`: The angle offset in the gate that we wish to measure.
        - :math:`{\rm phase}`: Phase of the oscillation.

    Initial Guesses
        - :math:`amp`: The maximum y value less the minimum y value.
        - :math:`baseline`: The average of the data.
        - :math:`{\rm d}\theta`: Zero.
        - :math:`{\rm phase}`: Zero.

    Bounds
        - :math:`amp`: [-1, 1] scaled to the maximum signal value.
        - :math:`baseline`: [-1, 1] scaled to the maximum signal value.
        - :math:`{\rm d}\theta`: [-pi, pi].
        - :math:`{\rm phase}`: [-pi, pi].
    """

    # TODO This will only work for pi pulses.
    __series__ = [
        SeriesDef(
            fit_func=lambda x, amp, d_theta, phase, baseline: fit_function.cos(
                x, amp=0.5 * amp, freq=d_theta + np.pi, phase=phase, baseline=baseline
            ),
            plot_color="blue",
        )
    ]

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.p0 = {"amp": None, "d_theta": None, "phase": None, "baseline": None}
        default_options.bounds = {"amp": None, "d_theta": None, "phase": None, "baseline": None}
        default_options.fit_reports = {"d_theta": "d_theta"}
        default_options.xlabel = "Number of gates (n)"
        default_options.ylabel = "Population"

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        b_guess = np.average(self._data().y)
        a_guess = np.max(self._data().y) - np.min(self._data().y) - b_guess

        max_abs_y = np.max(np.abs(self._data().y))

        # Base the initial guess on the intended angle_per_gate.
        # TODO This is hardcoded for now until CurveAnalysis can accept fix_parameters
        angle_per_gate = np.pi

        if angle_per_gate == 0:
            guess_range = np.pi / 2
        else:
            guess_range = angle_per_gate / 2

        fit_options = []

        for angle in np.linspace(-guess_range, guess_range, 11):
            fit_option = {
                "p0": {
                    "amp": user_p0["amp"] or a_guess,
                    "d_theta": angle,
                    "phase": 0.0,
                    "baseline": b_guess,
                },
                "bounds": {
                    "amp": user_bounds["amp"] or (-2 * max_abs_y, 2 * max_abs_y),
                    "d_theta": user_bounds["d_theta"] or (-np.pi, np.pi),
                    "phase": user_bounds["phase"] or (-np.pi, np.pi),
                    "baseline":  user_bounds["d_theta"] or (-1 * max_abs_y, 1 * max_abs_y),
                }
            }

            fit_options.append(fit_option)

        return fit_options

    def _post_analysis(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - an error on the measured angle deviation that is lower than the measured value.
        """
        fit_d_theta = get_opt_value(analysis_result, "d_theta")
        fit_d_theta_err = get_opt_error(analysis_result, "d_theta")

        criteria = [
            analysis_result["reduced_chisq"] < 3,
            (fit_d_theta_err is None or (fit_d_theta_err < fit_d_theta)),
        ]

        if all(criteria):
            analysis_result["quality"] = "computer_good"
        else:
            analysis_result["quality"] = "computer_bad"

        return analysis_result
