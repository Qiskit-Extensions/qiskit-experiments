# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Analyze oscillating data of cross resonance Rabi."""


from typing import List, Union

import lmfit
import numpy as np

from qiskit_experiments.framework import Options
import qiskit_experiments.curve_analysis as curve


class CrossResRabiAnalysis(curve.CurveAnalysis):
    r"""Cross resonance Rabi oscillation analysis class with nonlinear frequency.

    # section: fit_model

        Under the perturbation approximation, the amplitude dependence of
        the controlled rotation term, i.e. :math:`ZX` term,
        in the cross resonance Hamiltonian might be fit by [1]

        .. math::

            y = {\rm amp} \cos\left(
                2 \pi\cdot \left( {\rm freq}^{o1} \cdot x + {\rm freq}^{o3} \cdot x^3 \right) + \pi
                \right) + {\rm base}

        This approximation is valid as long as the tone amplitude is sufficiently weaker
        than the breakdown point at :math:`\Omega/\Delta \ll 1` where
        :math:`\Omega` is the tone amplitude and :math:`\Delta` is the qubit-qubit detuning.

    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of the oscillation.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.guess.max_height`.
            bounds: [0, 1]

        defpar \rm base:
            desc: Base line.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
            guess.constant_sinusoidal_offset`.
            bounds: [-1, 1]

        defpar \rm freq_o1:
            desc: Frequency of the oscillation in the first order.
                This is the fit parameter of interest.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.guess.frequency`.
            bounds: [0, inf].

        defpar \rm freq_o3:
            desc: Frequency of the oscillation in the third order.
                This is the fit parameter of interest.
            init_guess: 0.0.
            bounds: [0, inf].

    # section: reference
        .. ref_arxiv:: 1 1804.04073

    """

    def __init__(self):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * cos(2 * pi * (freq_o1 * x + freq_o3 * x**3) + pi) + offset",
                    name="cos",
                )
            ],
        )

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.update_options(
            result_parameters=[
                curve.ParameterRepr("freq_o1", "cross_res_rabi_rate_o1"),
                curve.ParameterRepr("freq_o3", "cross_res_rabi_rate_o3"),
            ],
            outcome="1",
        )
        options.plotter.set_figure_options(xlabel="Amplitude", ylabel="Target P(1)")
        return options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        y_offset = curve.guess.constant_sinusoidal_offset(curve_data.y)

        user_opt.bounds.set_if_empty(
            amp=(0, 1),
            freq_o1=(0, np.inf),
            freq_o3=(-np.inf, np.inf),
            offset=[-1, 1],
        )
        user_opt.p0.set_if_empty(
            offset=y_offset,
        )
        user_opt.p0.set_if_empty(
            freq_o1=curve.guess.frequency(curve_data.x, curve_data.y - user_opt.p0["offset"]),
            freq_o3=0.0,
            amp=curve.guess.max_height(curve_data.y - user_opt.p0["offset"], absolute=True)[0],
        )

        return user_opt

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - more than a quarter of a full period,
            - an error on the fit frequency lower than the fit frequency.
        """
        fit_freq_o1 = fit_data.ufloat_params["freq_o1"]
        fit_freq_o3 = fit_data.ufloat_params["freq_o3"]

        criteria = [
            curve.utils.is_error_not_significant(fit_freq_o1),
            curve.utils.is_error_not_significant(fit_freq_o3),
        ]

        if all(criteria):
            return "good"

        return "bad"
