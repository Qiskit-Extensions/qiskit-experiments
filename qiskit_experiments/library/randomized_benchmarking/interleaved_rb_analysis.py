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
Interleaved RB analysis class.
"""
from typing import List, Dict, Any, Union

import numpy as np

from qiskit_experiments.curve_analysis import (
    SeriesDef,
    CurveAnalysisResultData,
    fit_function,
    get_opt_value,
    get_opt_error,
)
from .rb_analysis import RBAnalysis


class InterleavedRBAnalysis(RBAnalysis):
    r"""A class to analyze interleaved randomized benchmarking experiment.

    # section: overview
        This analysis takes only two series for standard and interleaved RB curve fitting.
        From the fit :math:`\alpha` and :math:`\alpha_c` value this analysis estimates
        the error per Clifford (EPC) of the interleaved gate.

        The EPC estimate is obtained using the equation

        .. math::

            r_{\mathcal{C}}^{\text{est}} =
                \frac{\left(d-1\right)\left(1-\alpha_{\overline{\mathcal{C}}}/\alpha\right)}{d}

        The systematic error bounds are given by

        .. math::

            E = \min\left\{
                \begin{array}{c}
                    \frac{\left(d-1\right)\left[\left|\alpha-\alpha_{\overline{\mathcal{C}}}\right|
                    +\left(1-\alpha\right)\right]}{d} \\
                    \frac{2\left(d^{2}-1\right)\left(1-\alpha\right)}
                    {\alpha d^{2}}+\frac{4\sqrt{1-\alpha}\sqrt{d^{2}-1}}{\alpha}
                \end{array}
            \right.

        See Ref. [1] for more details.

    # section: fit_model
        The fit is based on the following decay functions:

        Fit model for standard RB

        .. math::

            F(x) = a \alpha^{x} + b

        Fit model for interleaved RB

        .. math::

            F(x) = a (\alpha_c \alpha)^{x_2} + b

    # section: fit_parameters
        defpar a:
            desc: Height of decay curve.
            init_guess: Determined by the average :math:`a` of the standard and interleaved RB.
            bounds: [0, 1]
        defpar b:
            desc: Base line.
            init_guess: Determined by the average :math:`b` of the standard and interleaved RB.
                Usually equivalent to :math:`(1/2)^n` where :math:`n` is number of qubit.
            bounds: [0, 1]
        defpar \alpha:
            desc: Depolarizing parameter.
            init_guess: Determined by the slope of :math:`(y - b)^{-x}` of the first and the
                second data point of the standard RB.
            bounds: [0, 1]
        defpar \alpha_c:
            desc: Ratio of the depolarizing parameter of interleaved RB to standard RB curve.
            init_guess: Estimate :math:`\alpha' = \alpha_c \alpha` from the
                interleaved RB curve, then divide this by the initial guess of :math:`\alpha`.
            bounds: [0, 1]

    # section: reference
        .. ref_arxiv:: 1 1203.4550

    """

    __series__ = [
        SeriesDef(
            name="Standard",
            fit_func=lambda x, a, alpha, alpha_c, b: fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha, baseline=b
            ),
            filter_kwargs={"interleaved": False},
            plot_color="red",
            plot_symbol=".",
            plot_fit_uncertainty=True,
        ),
        SeriesDef(
            name="Interleaved",
            fit_func=lambda x, a, alpha, alpha_c, b: fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha * alpha_c, baseline=b
            ),
            filter_kwargs={"interleaved": True},
            plot_color="orange",
            plot_symbol="^",
            plot_fit_uncertainty=True,
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        default_options = super()._default_options()
        default_options.p0 = {"a": None, "alpha": None, "alpha_c": None, "b": None}
        default_options.bounds = {
            "a": (0.0, 1.0),
            "alpha": (0.0, 1.0),
            "alpha_c": (0.0, 1.0),
            "b": (0.0, 1.0),
        }
        default_options.fit_reports = {"alpha": "\u03B1", "alpha_c": "\u03B1$_c$", "EPC": "EPC"}

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        # for standard RB curve
        std_curve = self._data(series_name="Standard")
        p0_std = self._initial_guess(std_curve.x, std_curve.y, self._num_qubits)

        # for interleaved RB curve
        int_curve = self._data(series_name="Interleaved")
        p0_int = self._initial_guess(int_curve.x, int_curve.y, self._num_qubits)

        fit_option = {
            "p0": {
                "a": user_p0["a"] or np.mean([p0_std["a"], p0_int["a"]]),
                "alpha": user_p0["alpha"] or p0_std["alpha"],
                "alpha_c": user_p0["alpha_c"] or min(p0_int["alpha"] / p0_std["alpha"], 1),
                "b": user_p0["b"] or np.mean([p0_std["b"], p0_int["b"]]),
            },
            "bounds": {
                "a": user_bounds["a"] or (0.0, 1.0),
                "alpha": user_bounds["alpha"] or (0.0, 1.0),
                "alpha_c": user_bounds["alpha_c"] or (0.0, 1.0),
                "b": user_bounds["b"] or (0.0, 1.0),
            },
        }
        fit_option.update(options)

        return fit_option

    def _post_analysis(self, result_data: CurveAnalysisResultData) -> CurveAnalysisResultData:
        """Calculate EPC."""
        # Add EPC data
        nrb = 2 ** self._num_qubits
        scale = (nrb - 1) / nrb
        alpha = get_opt_value(result_data, "alpha")
        alpha_c = get_opt_value(result_data, "alpha_c")
        alpha_c_err = get_opt_error(result_data, "alpha_c")

        # Calculate epc_est (=r_c^est) - Eq. (4):
        epc_est = scale * (1 - alpha_c)
        epc_est_err = scale * alpha_c_err
        result_data["EPC"] = epc_est
        result_data["EPC_err"] = epc_est_err

        # Calculate the systematic error bounds - Eq. (5):
        systematic_err_1 = scale * (abs(alpha - alpha_c) + (1 - alpha))
        systematic_err_2 = (
            2 * (nrb * nrb - 1) * (1 - alpha) / (alpha * nrb * nrb)
            + 4 * (np.sqrt(1 - alpha)) * (np.sqrt(nrb * nrb - 1)) / alpha
        )
        systematic_err = min(systematic_err_1, systematic_err_2)
        systematic_err_l = epc_est - systematic_err
        systematic_err_r = epc_est + systematic_err
        result_data["EPC_systematic_err"] = systematic_err
        result_data["EPC_systematic_bounds"] = [max(systematic_err_l, 0), systematic_err_r]

        return result_data
