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

"""DRAG pulse calibration experiment."""

from typing import List, Union

import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.fit_function import cos


class DragCalAnalysis(curve.CurveAnalysis):
    r"""Drag calibration analysis based on a fit to a cosine function.

    # section: fit_model

        Analyse a Drag calibration experiment by fitting three series each to a cosine
        function. The three functions share the phase parameter (i.e. beta), amplitude, and
        baseline. The frequencies of the oscillations are related through the number of
        repetitions of the Drag gates. Several initial guesses are tried if the user
        does not provide one.

        .. math::

            y_i = {\rm amp} \cos\left(2 \pi\cdot {\rm freq}_i\cdot x -
            2 \pi\cdot {\rm freq}_i\cdot \beta\right) + {\rm base}

        Here, :math:`{\rm freq}_i` is given by the fit parameter :math:`freq` multiplied by
        :math:`{\rm rep}_i` which is the number of times that the Drag plus and minus
        rotations are repeated. Note that the aim of the Drag calibration is to find the
        :math:`\beta` that minimizes the phase shifts. This implies that the optimal
        :math:`\beta` occurs when all three :math:`y` curves are minimum, i.e. they
        produce the ground state. Therefore,

        .. math::

            y_i = 0 \quad \Longrightarrow \quad -{\rm amp} \cos(2 \pi\cdot X_i) = {\rm base}

        Here, we abbreviated :math:`{\rm freq}_i\cdot x - {\rm freq}_i\cdot \beta` by :math:`X_i`.
        For a signal between 0 and 1 the :math:`{\rm base}` will typically fit to 0.5. However, the
        equation has an ambiguity if the amplitude is not properly bounded. Indeed,

        - if :math:`{\rm amp} < 0` then we require :math:`2 \pi\cdot X_i = 0` mod :math:`2\pi`, and
        - if :math:`{\rm amp} > 0` then we require :math:`2 \pi\cdot X_i = \pi` mod :math:`2\pi`.

        This will result in an ambiguity in :math:`\beta` which we avoid by bounding the amplitude
        from above by 0.

    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of all series.
            init_guess: The maximum y value less the minimum y value scaled by -1, -0.5, and -0.25.
            bounds: [-2, 0] scaled to the maximum signal value.

        defpar \rm base:
            desc: Base line of all series.
            init_guess: Half the maximum y-range of the data.
            bounds: [-1, 1] scaled to the maximum y-range.

        defpar {\rm freq}:
            desc: Frequency of the :math:`i` th oscillation divided by the number of repetitions.
            init_guess: The frequency with the highest power spectral density of the curve with
                the highest number of Drag plus and minus rotation repetitions.
            bounds: [0, inf].

        defpar \beta:
            desc: Common beta offset. This is the parameter of interest.
            init_guess: Linearly spaced between the maximum and minimum scanned beta.
            bounds: [-min scan range, max scan range].
    """

    __series__ = [
        curve.SeriesDef(
            fit_func=lambda x, amp, freq, reps1, reps2, reps3, beta, base: cos(
                x, amp=amp, freq=reps1 * freq, phase=-2 * np.pi * reps1 * freq * beta, baseline=base
            ),
            plot_color="blue",
            name="series-0",
            filter_kwargs={"series": 0},
            plot_symbol="o",
            model_description=r"{\rm amp} \cos\left(2 \pi\cdot {\rm freq}_0\cdot x "
            r"- 2 \pi\cdot {\rm freq}_0\cdot \beta\right) + {\rm base}",
        ),
        curve.SeriesDef(
            fit_func=lambda x, amp, freq, reps1, reps2, reps3, beta, base: cos(
                x, amp=amp, freq=reps2 * freq, phase=-2 * np.pi * reps2 * freq * beta, baseline=base
            ),
            plot_color="green",
            name="series-1",
            filter_kwargs={"series": 1},
            plot_symbol="^",
            model_description=r"{\rm amp} \cos\left(2 \pi\cdot {\rm freq}_1\cdot x "
            r"- 2 \pi\cdot {\rm freq}_1\cdot \beta\right) + {\rm base}",
        ),
        curve.SeriesDef(
            fit_func=lambda x, amp, freq, reps1, reps2, reps3, beta, base: cos(
                x, amp=amp, freq=reps3 * freq, phase=-2 * np.pi * reps3 * freq * beta, baseline=base
            ),
            plot_color="red",
            name="series-2",
            filter_kwargs={"series": 2},
            plot_symbol="v",
            model_description=r"{\rm amp} \cos\left(2 \pi\cdot {\rm freq}_2\cdot x "
            r"- 2 \pi\cdot {\rm freq}_2\cdot \beta\right) + {\rm base}",
        ),
    ]

    __fixed_parameters__ = ["reps1", "reps2", "reps3"]

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.result_parameters = ["beta"]
        default_options.xlabel = "Beta"
        default_options.ylabel = "Signal (arb. units)"
        default_options.rep1 = 1
        default_options.rep2 = 3
        default_options.rep3 = 5

        return default_options

    def _generate_fit_guesses(
        self, user_opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        # Use a fast Fourier transform to guess the frequency.
        x_data = self._data("series-0").x
        min_beta, max_beta = min(x_data), max(x_data)

        # Use the highest-frequency curve to estimate the oscillation frequency.
        curve_data = self._data(f"series-2")
        freqs_guess = curve.guess.frequency(curve_data.x, curve_data.y) / self.options.reps3
        user_opt.p0.set_if_empty(freq=freqs_guess)

        max_abs_y, _ = curve.guess.max_height(self._data().y, absolute=True)
        freq_bound = max(10 / user_opt.p0["freq"], max(x_data))

        user_opt.bounds.set_if_empty(
            amp=(-2 * max_abs_y, 0),
            freq=(0, np.inf),
            beta=(-freq_bound, freq_bound),
            base=(-max_abs_y, max_abs_y),
        )
        user_opt.p0.set_if_empty(base=(user_opt.p0["amp"] or max_abs_y) / 2)

        # Drag curves can sometimes be very flat, i.e. averages of y-data
        # and min-max do not always make good initial guesses. We therefore add
        # 0.5 to the initial guesses. Note that we also set amp=-0.5 because the cosine function
        # becomes +1 at zero phase, i.e. optimal beta, in which y data should become zero
        # in discriminated measurement level.
        options = []
        for amp_factor in (-1, -0.5, -0.25):
            for beta_guess in np.linspace(min_beta, max_beta, 20):
                new_opt = user_opt.copy()
                new_opt.p0.set_if_empty(amp=max_abs_y * amp_factor, beta=beta_guess)
                options.append(new_opt)

        return options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - a DRAG parameter value within the first period of the lowest number of repetitions,
            - an error on the drag beta smaller than the beta.
        """
        fit_beta = fit_data.fitval("beta")
        fit_freq = fit_data.fitval("freq")

        criteria = [
            fit_data.reduced_chisq < 3,
            fit_beta.nominal_value < 1 / fit_freq.nominal_value,
            curve.is_error_not_significant(fit_beta),
        ]

        if all(criteria):
            return "good"

        return "bad"
