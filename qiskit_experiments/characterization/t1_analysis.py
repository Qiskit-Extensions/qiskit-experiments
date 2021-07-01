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

from typing import Tuple, List
import numpy as np

from qiskit.providers.options import Options
from qiskit.utils import apply_prefix

from qiskit_experiments.experiment_data import AnalysisResult
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.analysis.curve_fitting import process_curve_data, curve_fit
from qiskit_experiments.analysis.data_processing import level2_probability
from qiskit_experiments.analysis import plotting


class T1Analysis(BaseAnalysis):
    """T1 Experiment result analysis class.

    Analysis Options:

        * t1_guess (float): Optional, an initial guess of T1.
        * amplitude_guess (float): Optional, an initial guess of the
                                   coefficient of the exponent.
        * offset_guess (float): Optional, an initial guess of the offset.
        * t1_bounds (list of two floats): Optional, lower bound and upper
                                          bound to T1.
        * amplitude_bounds (list of two floats): Optional, lower bound and upper
                                                 bound to the amplitude.
        * offset_bounds (list of two floats): Optional, lower bound and
                                              upper bound to the offset.
    """

    @classmethod
    def _default_options(cls):
        return Options(
            t1_guess=None,
            amplitude_guess=None,
            offset_guess=None,
            t1_bounds=None,
            amplitude_bounds=None,
            offset_bounds=None,
        )

    # pylint: disable=arguments-differ
    def _run_analysis(
        self,
        experiment_data,
        t1_guess=None,
        amplitude_guess=None,
        offset_guess=None,
        t1_bounds=None,
        amplitude_bounds=None,
        offset_bounds=None,
        plot=True,
        ax=None,
    ) -> Tuple[List[AnalysisResult], List["matplotlib.figure.Figure"]]:
        """
        Calculate T1

        Args:
            experiment_data (ExperimentData): the experiment data to analyze
            t1_guess (float): Optional, an initial guess of T1
            amplitude_guess (float): Optional, an initial guess of the coefficient
                                     of the exponent
            offset_guess (float): Optional, an initial guess of the offset
            t1_bounds (list of two floats): Optional, lower bound and upper bound to T1
            amplitude_bounds (list of two floats): Optional, lower bound and upper
                                                   bound to the amplitude
            offset_bounds (list of two floats): Optional, lower bound and upper
                                                bound to the offset
            plot (bool): Generator plot of exponential fit.
            ax (AxesSubplot): Optional, axes to add figure to.

        Returns:
            The analysis result with the estimated T1

        Raises:
            AnalysisError: if the analysis fails.
        """
        data = experiment_data.data()
        unit = data[0]["metadata"]["unit"]
        conversion_factor = data[0]["metadata"].get("dt_factor", None)
        qubit = data[0]["metadata"]["qubit"]

        if conversion_factor is None:
            conversion_factor = 1 if unit == "s" else apply_prefix(1, unit)

        xdata, ydata, sigma = process_curve_data(data, lambda datum: level2_probability(datum, "1"))
        xdata *= conversion_factor

        if t1_guess is None:
            t1_guess = np.mean(xdata)
        else:
            t1_guess = t1_guess * conversion_factor
        if offset_guess is None:
            offset_guess = ydata[-1]
        if amplitude_guess is None:
            amplitude_guess = ydata[0] - offset_guess
        if t1_bounds is None:
            t1_bounds = [0, np.inf]
        if amplitude_bounds is None:
            amplitude_bounds = [0, 1]
        if offset_bounds is None:
            offset_bounds = [0, 1]

        # Perform fit
        def fit_fun(x, a, tau, c):
            return a * np.exp(-x / tau) + c

        init = {"a": amplitude_guess, "tau": t1_guess, "c": offset_guess}
        bounds = {"a": amplitude_bounds, "tau": t1_bounds, "c": offset_bounds}
        fit_result = curve_fit(fit_fun, xdata, ydata, init, sigma=sigma, bounds=bounds)

        analysis_result = AnalysisResult(
            {
                "value": fit_result["popt"][1],
                "stderr": fit_result["popt_err"][1],
                "unit": "s",
                "label": "T1",
                "fit": fit_result,
                "quality": self._fit_quality(
                    fit_result["popt"], fit_result["popt_err"], fit_result["reduced_chisq"]
                ),
            }
        )

        analysis_result["fit"]["circuit_unit"] = unit
        if unit == "dt":
            analysis_result["fit"]["dt"] = conversion_factor

        # Generate fit plot
        if plot and plotting.HAS_MATPLOTLIB:
            ax = plotting.plot_curve_fit(
                fit_fun, fit_result, monotonicity={"a": True, "tau": True, "c": True}, ax=ax
            )
            ax = plotting.plot_errorbar(xdata, ydata, sigma, ax=ax)
            self._format_plot(ax, fit_result, qubit=qubit)
            figures = [ax.get_figure()]
        else:
            figures = None

        return [analysis_result], figures

    @staticmethod
    def _fit_quality(fit_out, fit_err, reduced_chisq):
        # pylint: disable = too-many-boolean-expressions
        if (
            abs(fit_out[0] - 1.0) < 0.1
            and abs(fit_out[2]) < 0.1
            and reduced_chisq < 3
            and (fit_err[0] is None or fit_err[0] < 0.1)
            and (fit_err[1] is None or fit_err[1] < fit_out[1])
            and (fit_err[2] is None or fit_err[2] < 0.1)
        ):
            return "computer_good"
        else:
            return "computer_bad"

    @classmethod
    def _format_plot(cls, ax, analysis_result, qubit=None, add_label=True):
        """Format curve fit plot"""
        # Formatting
        ax.tick_params(labelsize=14)
        if qubit is not None:
            ax.set_title(f"Qubit {qubit}", fontsize=16)
        ax.set_xlabel("Delay (s)", fontsize=16)
        ax.set_ylabel("P(1)", fontsize=16)
        ax.grid(True)

        if add_label:
            t1 = analysis_result["popt"][1]
            t1_err = analysis_result["popt_err"][1]
            # Convert T1 to time unit for pretty printing
            if t1 < 1e-7:
                scale = 1e9
                unit = "ns"
            elif t1 < 1e-4:
                scale = 1e6
                unit = "μs"
            elif t1 < 0.1:
                scale = 1e3
                unit = "ms"
            else:
                scale = 1
                unit = "s"
            box_text = "$T_1$ = {:.2f} \u00B1 {:.2f} {}".format(t1 * scale, t1_err * scale, unit)
            bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1)
            ax.text(
                0.6,
                0.9,
                box_text,
                ha="center",
                va="center",
                size=14,
                bbox=bbox_props,
                transform=ax.transAxes,
            )
        return ax
