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

"""The analysis class for the Ramsey XY experiment."""

from typing import List, Tuple, Union

import lmfit
import numpy as np
from uncertainties import unumpy as unp

import qiskit_experiments.curve_analysis as curve
import qiskit_experiments.visualization as vis
from qiskit_experiments.framework import ExperimentData, AnalysisResultData
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX


class RamseyXYAnalysis(curve.CurveAnalysis):
    r"""Ramsey XY analysis based on a fit to a cosine function and a sine function.

    # section: fit_model

        Analyze a Ramsey XY experiment by fitting the X and Y series to a cosine and sine
        function, respectively. The two functions share the frequency and amplitude parameters.

        .. math::

            y_X = {\rm amp}e^{-x/\tau}\cos\left(2\pi\cdot{\rm freq}_i\cdot x\right) + {\rm base} \\
            y_Y = {\rm amp}e^{-x/\tau}\sin\left(2\pi\cdot{\rm freq}_i\cdot x\right) + {\rm base}

    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of both series.
            init_guess: Half of the maximum y value less the minimum y value. When the
                oscillation frequency is low, it uses an averaged difference of
                Ramsey X data - Ramsey Y data.
            bounds: [0, 2 * average y peak-to-peak]

        defpar \tau:
            desc: The exponential decay of the curve.
            init_guess: The initial guess is obtained by fitting an exponential to the
                square root of (X data)**2 + (Y data)**2.
            bounds: [0, inf]

        defpar \rm base:
            desc: Base line of both series.
            init_guess: Roughly the average of the data. When the oscillation frequency is low,
                it uses an averaged data of Ramsey Y experiment.
            bounds: [min y - average y peak-to-peak, max y + average y peak-to-peak]

        defpar \rm freq:
            desc: Frequency of both series. This is the parameter of interest.
            init_guess: The frequency with the highest power spectral density.
            bounds: [-inf, inf]

        defpar \rm phase:
            desc: Common phase offset.
            init_guess: 0
            bounds: [-pi, pi]
    """

    def __init__(self):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x / tau) * cos(2 * pi * freq * x + phase) + base",
                    name="X",
                ),
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x / tau) * sin(2 * pi * freq * x + phase) + base",
                    name="Y",
                ),
            ]
        )

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.data_subfit_map = {
            "X": {"series": "X"},
            "Y": {"series": "Y"},
        }
        default_options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="Signal (arb. units)",
            xval_unit="s",
        )
        default_options.result_parameters = ["freq"]

        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.ScatterTable,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        ramx_data = curve_data.get_subset_of("X")
        ramy_data = curve_data.get_subset_of("Y")

        # At very low frequency, y value of X (Y) curve stay at P=1.0 (0.5) for all x values.
        # Computing y peak-to-peak with combined data gives fake amplitude of 0.25.
        # Same for base, i.e. P=0.75 is often estimated in this case.
        full_y_ptp = np.ptp(curve_data.y)
        avg_y_ptp = 0.5 * (np.ptp(ramx_data.y) + np.ptp(ramy_data.y))
        max_y = np.max(curve_data.y)
        min_y = np.min(curve_data.y)

        user_opt.bounds.set_if_empty(
            amp=(0, full_y_ptp * 2),
            tau=(0, np.inf),
            base=(min_y - avg_y_ptp, max_y + avg_y_ptp),
            phase=(-np.pi, np.pi),
        )

        if avg_y_ptp < 0.5 * full_y_ptp:
            # When X and Y curve don't oscillate, X (Y) usually stays at P(1) = 1.0 (0.5).
            # So peak-to-peak of full data is something around P(1) = 0.75, while
            # single curve peak-to-peak is almost zero.
            avg_x = np.average(ramx_data.y)
            avg_y = np.average(ramy_data.y)

            user_opt.p0.set_if_empty(
                amp=np.abs(avg_x - avg_y),
                tau=100 * np.max(curve_data.x),
                base=avg_y,
                phase=0.0,
                freq=0.0,
            )
            return user_opt

        base_guess_x = curve.guess.constant_sinusoidal_offset(ramx_data.y)
        base_guess_y = curve.guess.constant_sinusoidal_offset(ramy_data.y)
        base_guess = 0.5 * (base_guess_x + base_guess_y)
        user_opt.p0.set_if_empty(
            amp=0.5 * full_y_ptp,
            base=base_guess,
            phase=0.0,
        )

        # Guess the exponential decay by combining both curves
        ramx_unbiased = ramx_data.y - user_opt.p0["base"]
        ramy_unbiased = ramy_data.y - user_opt.p0["base"]
        decay_data = ramx_unbiased**2 + ramy_unbiased**2
        if np.ptp(decay_data) < 0.95 * 0.5 * full_y_ptp:
            # When decay is less than 95 % of peak-to-peak value, ignore decay and
            # set large enough tau value compared with the measured x range.
            user_opt.p0.set_if_empty(tau=1000 * np.max(curve_data.x))
        else:
            user_opt.p0.set_if_empty(tau=-1 / curve.guess.exp_decay(ramx_data.x, decay_data))

        # Guess the oscillation frequency, remove offset to eliminate DC peak
        freq_guess_x = curve.guess.frequency(ramx_data.x, ramx_unbiased)
        freq_guess_y = curve.guess.frequency(ramy_data.x, ramy_unbiased)
        freq_val = 0.5 * (freq_guess_x + freq_guess_y)

        # FFT might be up to 1/2 bin off
        df = 2 * np.pi / (np.min(np.diff(ramx_data.x)) * ramx_data.x.size)
        freq_guesses = [freq_val - df, freq_val + df, freq_val]

        # Ramsey XY is frequency sign sensitive.
        # Since experimental data is noisy, correct sign is hardly estimated with phase velocity.
        # Try both positive and negative frequency to find the best fit.
        opts = []
        for sign in (1, -1):
            for freq_guess in freq_guesses:
                opt = user_opt.copy()
                opt.p0.set_if_empty(freq=sign * freq_guess)
                opts.append(opt)

        return opts

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three and greater than zero,
            - an error on the frequency smaller than the frequency.
        """
        fit_freq = fit_data.ufloat_params["freq"]

        criteria = [
            0 < fit_data.reduced_chisq < 3,
            curve.utils.is_error_not_significant(fit_freq),
        ]

        if all(criteria):
            return "good"

        return "bad"


class StarkRamseyXYAmpScanAnalysis(curve.CurveAnalysis):
    r"""Ramsey XY analysis for the Stark shifted phase sweep.

    # section: overview

        This analysis is a variant of :class:`RamseyXYAnalysis` in which
        the data is fit for a trigonometric function model with a linear phase.
        By contrast, in this model, the phase is assumed to be a polynomial of
        the x-data :math:`\theta(x)`, and techniques to compute a good initial guess
        for these polynomial coefficients inside the trignometric function are not trivial.
        Instead, this analysis performs heavy data formatting to extract
        raw phase polynomial :math:`\theta(x)` and run curve fitting on the synthesized data.

        The measured P1 values for Ramsey X and Y experiment can be written in a form of
        a trignometric function taking the phase polynomial :math:`\theta(x)`:

        .. math::

            P_X =  \text{amp}(x) \cdot \cos \theta(x) + \text{offset},\\
            P_Y =  \text{amp}(x) \cdot \sin \theta(x) + \text{offset}.

        Hence the phase polynomial can be extracted as follows

        .. math::

            \theta(x) = \tan^{-1} \frac{P_Y}{P_X}.

        Because the arctangent is implemented by the ``atan2`` function
        defined in :math:`[-\pi, \pi]`, the computed :math:`\theta(x)` is unwrapped to
        ensure the continuous phase evolution.

        We call attantion to the fact that :math:`\text{amp}(x)` is also Stark tone amplitude
        dependent because of the qubit frequency dependence of the dephasing rate.
        In general :math:`\text{amp}(x)` is unpredictable due to random occurence of TLS
        or probably due to qubit heating, and this prevents us from precisely fitting
        the raw :math:`P_X`, :math:`P_Y` data. Fitting on the polynomial makes the
        analysis robust to the amplitude dependent dephasing.

        In this analysis, the phase polynomial is defined as

        .. math::

            \theta(x) = 2 \pi t_S f_S(x)

        where

        .. math::

            f_S(x) = c_1 x + c_2 x^2 + c_3 x^3 + f_{\rm err},

        denotes the Stark shift. For the lowest order perturbative expansion of a single driven qubit,
        the Stark shift is a quadratic function of :math:`x`, but linear and cubic terms
        and a constant offset are also considered to account for
        other effects, e.g. strong drive, collisions, TLS, and so forth,
        and frequency mis-calibration, respectively.

    # section: fit_model

        .. math::

            \theta^\nu(x) = 2 \pi t_S \left(
                c_1^\nu x + c_2^\nu x^2 + c_3^\nu x^3 + f_{\rm err} \right),

        where :math:`\nu \in \{+, -\}`.
        The Stark shift is asymmetric with respect to :math:`x=0`, because of the
        anti-crossings of higher energy levels. In a typical transmon qubit,
        these levels appear only in :math:`f_S < 0` because of the negative anharmonicity.
        To precisely fit the results, this analysis uses different model parameters
        for positive (:math:`x > 0`) and negative (:math:`x < 0`) shift domains.

    # section: fit_parameters

        defpar t_S:
            desc: Fixed parameter from the ``stark_length`` experiment option.
            init_guess: Automatically set from metadata when this analysis is run.
            bounds: None

        defpar c_1^+:
            desc: The linear term coefficient of the positive Stark shift
                (fit parameter: ``stark_pos_coef_o1``).
            init_guess: 0.
            bounds: None

        defpar c_2^+:
            desc: The quadratic term coefficient of the positive Stark shift.
                This parameter must be positive because Stark amplitude is chosen to
                induce blue shift when its sign is positive.
                Note that the quadratic term is the primary term
                (fit parameter: ``stark_pos_coef_o2``).
            init_guess: 1.
            bounds: [0, inf]

        defpar c_3^+:
            desc: The cubic term coefficient of the positive Stark shift
                (fit parameter: ``stark_pos_coef_o3``).
            init_guess: 0.
            bounds: None

        defpar c_1^-:
            desc: The linear term coefficient of the negative Stark shift.
                (fit parameter: ``stark_neg_coef_o1``).
            init_guess: 0.
            bounds: None

        defpar c_2^-:
            desc: The quadratic term coefficient of the negative Stark shift.
                This parameter must be negative because Stark amplitude is chosen to
                induce red shift when its sign is negative.
                Note that the quadratic term is the primary term
                (fit parameter: ``stark_neg_coef_o2``).
            init_guess: -1.
            bounds: [-inf, 0]

        defpar c_3^-:
            desc: The cubic term coefficient of the negative Stark shift
                (fit parameter: ``stark_neg_coef_o3``).
            init_guess: 0.
            bounds: None

        defpar f_{\rm err}:
            desc: Constant phase accumulation which is independent of the Stark tone amplitude.
                (fit parameter: ``stark_ferr``).
            init_guess: Averaege of y values at minimum absolute x values on
                positive and negative shift data.
            bounds: None

    # section: see_also

        :class:`qiskit_experiments.library.characterization.analysis.ramsey_xy_analysis.RamseyXYAnalysis`

    """

    def __init__(self):

        models = [
            lmfit.models.ExpressionModel(
                expr="2 * pi * ts * (c1_pos * x + c2_pos * x**2 + c3_pos * x**3 + f_err)",
                name="Fpos",
            ),
            lmfit.models.ExpressionModel(
                expr="2 * pi * ts * (c1_neg * x + c2_neg * x**2 + c3_neg * x**3 + f_err)",
                name="Fneg",
            ),
        ]
        super().__init__(models=models)

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        ramsey_plotter = vis.CurvePlotter(vis.MplDrawer())
        ramsey_plotter.set_figure_options(
            xlabel="Stark tone amplitude",
            ylabel=["Stark shift", "P1"],
            yval_unit=["Hz", None],
            series_params={
                "Fpos": {
                    "color": "#123FE8",
                    "symbol": "^",
                    "label": "",
                    "canvas": 0,
                },
                "Fneg": {
                    "color": "#123FE8",
                    "symbol": "v",
                    "label": "",
                    "canvas": 0,
                },
                "Xpos": {
                    "color": "#123FE8",
                    "symbol": "o",
                    "label": "Ramsey X",
                    "canvas": 1,
                },
                "Ypos": {
                    "color": "#6312E8",
                    "symbol": "^",
                    "label": "Ramsey Y",
                    "canvas": 1,
                },
                "Xneg": {
                    "color": "#E83812",
                    "symbol": "o",
                    "label": "Ramsey X",
                    "canvas": 1,
                },
                "Yneg": {
                    "color": "#E89012",
                    "symbol": "^",
                    "label": "Ramsey Y",
                    "canvas": 1,
                },
            },
            sharey=False,
        )
        ramsey_plotter.set_options(subplots=(2, 1), style=vis.PlotStyle({"figsize": (10, 8)}))

        options = super()._default_options()
        options.update_options(
            result_parameters=[
                curve.ParameterRepr("c1_pos", "stark_pos_coef_o1", "Hz"),
                curve.ParameterRepr("c2_pos", "stark_pos_coef_o2", "Hz"),
                curve.ParameterRepr("c3_pos", "stark_pos_coef_o3", "Hz"),
                curve.ParameterRepr("c1_neg", "stark_neg_coef_o1", "Hz"),
                curve.ParameterRepr("c2_neg", "stark_neg_coef_o2", "Hz"),
                curve.ParameterRepr("c3_neg", "stark_neg_coef_o3", "Hz"),
                curve.ParameterRepr("f_err", "stark_ferr", "Hz"),
            ],
            data_subfit_map={
                "Xpos": {"series": "X", "direction": "pos"},
                "Ypos": {"series": "Y", "direction": "pos"},
                "Xneg": {"series": "X", "direction": "neg"},
                "Yneg": {"series": "Y", "direction": "neg"},
            },
            plotter=ramsey_plotter,
        )

        return options

    def _to_phase_data(
        self,
        curve_data: curve.CurveData,
    ) -> curve.CurveData:
        """Convert Ramsey XY data into frequency shift by computing the phase evolution.

        Args:
            curve_data: Processed dataset created from experiment results.
                This data must include four series of Xpos, Xneg, Ypos, and Yneg.
                Y values are measured P1.

        Returns:
            Formatted data. This data includes two series of Fpos and Fneg.
            Y values are frequencies in Hz.
        """
        y_mean = np.mean(curve_data.y)

        new_xs = []
        new_ys = []
        new_yerrs = []
        new_shots = []
        for direction in ("pos", "neg"):
            x_quadrature = curve_data.get_subset_of(f"X{direction}")
            y_quadrature = curve_data.get_subset_of(f"Y{direction}")

            if not np.array_equal(x_quadrature.x, y_quadrature.x):
                raise ValueError(
                    "Amplitude values of X and Y quadrature are different. Same values must be used."
                )
            xq_uarray = unp.uarray(x_quadrature.y, x_quadrature.y_err)
            yq_uarray = unp.uarray(y_quadrature.y, y_quadrature.y_err)

            amplitudes = x_quadrature.x
            shots = x_quadrature.shots + y_quadrature.shots

            # pylint: disable=no-member
            phase = unp.arctan2(yq_uarray - y_mean, xq_uarray - y_mean)
            phase_n = unp.nominal_values(phase)
            phase_s = unp.std_devs(phase)

            # Unwrap phase
            # We assume a smooth slope and correct 2pi phase jump to minimize the change of the slope.
            if amplitudes[0] < 0:
                # Flip array order because this is array is negative increments.
                phase_n = phase_n[::-1]
            unwrapped_phase = np.unwrap(phase_n)
            if amplitudes[0] < 0:
                # Flip back
                unwrapped_phase = unwrapped_phase[::-1]
            # Store new data
            new_xs.append(amplitudes)
            new_ys.append(unwrapped_phase)
            new_yerrs.append(phase_s)
            new_shots.append(shots)

        curve_data = curve.CurveData(
            x=np.concatenate(new_xs),
            y=np.concatenate(new_ys),
            y_err=np.concatenate(new_yerrs),
            shots=np.concatenate(new_shots),
            data_allocation=np.concatenate(([0] * len(new_xs[0]), [1] * len(new_xs[1]))),
            labels=["Fpos", "Fneg"],
        )
        return curve_data

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.ScatterTable,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.bounds.set_if_empty(c2_pos=(0, np.inf), c2_neg=(-np.inf, 0))

        pos_y = curve_data.get_subset_of("Fpos").y
        neg_y = curve_data.get_subset_of("Fneg").y

        user_opt.p0.set_if_empty(
            c1_pos=0,
            c2_pos=1,
            c3_pos=0,
            c1_neg=0,
            c2_neg=-1,
            c3_neg=0,
            f_err=(pos_y[0] + neg_y[0]) / 2,
        )
        return user_opt

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        super()._initialize(experiment_data)

        # Set scaling factor to convert phase to frequency
        fixed_params = self.options.fixed_parameters.copy()
        fixed_params["ts"] = experiment_data.metadata["stark_length"]
        self.set_options(fixed_parameters=fixed_params)

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:

        # TODO do not override this method.
        #  This phase fitter also plots raw Ramsey XY P1 curves.
        #  This requires generation of plot data for the second axis and current
        #  CurveAnalysis cannot handle this case with its pattern.
        #  We should split figure generation functionality from the curve fitting and
        #  make fit data portable for visualization in later stage.

        # Prepare for fitting
        self._initialize(experiment_data)

        analysis_results = []
        const = 2 * np.pi * experiment_data.metadata["stark_length"]

        # Run data processing
        processed_data = self._run_data_processing(
            raw_data=experiment_data.data(),
            models=self._models,
        )
        ramsey_xy_p1_data = self._format_data(processed_data)
        ramsey_xy_phase_data = self._to_phase_data(ramsey_xy_p1_data)

        # Plot raw data for phase and P1 in separate canvas
        if self.options.plot:
            for name in ("Fpos", "Fneg"):
                sub_data = ramsey_xy_phase_data.get_subset_of(name)
                self.plotter.set_series_data(
                    series_name=name,
                    x_formatted=sub_data.x,
                    y_formatted=sub_data.y / const,
                    y_formatted_err=sub_data.y_err / const,
                )
            for name in ("Xpos", "Ypos", "Xneg", "Yneg"):
                sub_data = ramsey_xy_p1_data.get_subset_of(name)
                self.plotter.set_series_data(
                    series_name=name,
                    x_formatted=sub_data.x,
                    y_formatted=sub_data.y,
                    y_formatted_err=sub_data.y_err,
                )

        # Run fitting
        fit_data = self._run_curve_fit(
            curve_data=ramsey_xy_phase_data,
            models=self._models,
        )

        if fit_data.success:
            quality = self._evaluate_quality(fit_data)
            self.plotter.set_supplementary_data(fit_red_chi=fit_data.reduced_chisq)
        else:
            quality = "bad"

        if self.options.return_fit_parameters:
            # Store fit status overview entry regardless of success.
            # This is sometime useful when debugging the fitting code.
            overview = AnalysisResultData(
                name=PARAMS_ENTRY_PREFIX + self.name,
                value=fit_data,
                quality=quality,
                extra=self.options.extra,
            )
            analysis_results.append(overview)

        # Create figure and result data
        if fit_data.success:
            # Create analysis results
            primary_results = self._create_analysis_results(
                fit_data=fit_data, quality=quality, **self.options.extra.copy()
            )
            analysis_results.extend(primary_results)
            self.plotter.set_supplementary_data(primary_results=primary_results)

            # Draw fit curves and report
            if self.options.plot:
                # Bootstrap model parameters for RamseyXY P1 plot
                offset_guess = np.mean(ramsey_xy_p1_data.y)
                amp_guess = np.max(np.abs((ramsey_xy_p1_data.y - offset_guess)))

                model_dict = {model._name: model for model in self._models}
                for model_suffix in ("pos", "neg"):
                    model_name = f"F{model_suffix}"
                    model = model_dict[model_name]
                    sub_data = ramsey_xy_phase_data.get_subset_of(model_name)
                    if sub_data.x.size == 0:
                        continue
                    x_interp = np.linspace(np.min(sub_data.x), np.max(sub_data.x), num=100)
                    y_data_with_uncertainty = curve.utils.eval_with_uncertainties(
                        x=x_interp,
                        model=model,
                        params=fit_data.ufloat_params,
                    )
                    y_interp = unp.nominal_values(y_data_with_uncertainty)
                    # Add fit line data
                    self.plotter.set_series_data(
                        model_name,
                        x_interp=x_interp,
                        y_interp=y_interp / const,
                    )
                    # Add confidence interval data
                    if fit_data.covar is not None:
                        y_interp_err = unp.std_devs(y_data_with_uncertainty)
                        if np.isfinite(y_interp_err).all():
                            self.plotter.set_series_data(
                                model_name,
                                y_interp_err=y_interp_err / const,
                            )
                    # Add second axis
                    # pylint: disable=no-member
                    ramsey_cos = amp_guess * unp.cos(y_data_with_uncertainty) + offset_guess
                    ramsey_sin = amp_guess * unp.sin(y_data_with_uncertainty) + offset_guess
                    self.plotter.set_series_data(
                        f"X{model_suffix}",
                        x_interp=x_interp,
                        y_interp=unp.nominal_values(ramsey_cos),
                    )
                    self.plotter.set_series_data(
                        f"Y{model_suffix}",
                        x_interp=x_interp,
                        y_interp=unp.nominal_values(ramsey_sin),
                    )
                    # Add confidence interval data to second axis
                    if fit_data.covar is not None:
                        if np.isfinite(y_interp_err).all():
                            self.plotter.set_series_data(
                                f"X{model_suffix}",
                                y_interp_err=unp.std_devs(ramsey_cos),
                            )
                            self.plotter.set_series_data(
                                f"Y{model_suffix}",
                                y_interp_err=unp.std_devs(ramsey_sin),
                            )

        # Add raw data points
        if self.options.return_data_points:
            analysis_results.extend(
                self._create_curve_data(curve_data=ramsey_xy_phase_data, models=self._models)
            )

        # Finalize plot
        if self.options.plot:
            return analysis_results, [self.plotter.figure()]

        return analysis_results, []
