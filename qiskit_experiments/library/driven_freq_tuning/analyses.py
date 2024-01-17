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
"""Stark shift analyses."""

from __future__ import annotations

from typing import List, Union

import lmfit
import numpy as np
from uncertainties import unumpy as unp

import qiskit_experiments.curve_analysis as curve
import qiskit_experiments.data_processing as dp
import qiskit_experiments.visualization as vis
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.framework import BaseAnalysis, ExperimentData, AnalysisResultData, Options
from .coefficient_utils import (
    StarkCoefficients,
    convert_amp_to_freq,
    retrieve_coefficients_from_service,
)


class StarkRamseyXYAmpScanAnalysis(curve.CurveAnalysis):
    r"""Ramsey XY analysis for the Stark shifted phase sweep.

    # section: overview

        This analysis is a variant of :class:`RamseyXYAnalysis`. In both cases, the X and Y
        data are treated as the real and imaginary parts of a complex oscillating signal.
        In :class:`RamseyXYAnalysis`, the data are fit assuming a phase varying linearly with
        the x-data corresponding to a constant frequency and assuming an exponentially
        decaying amplitude. By contrast, in this model, the phase is assumed to be
        a third order polynomial :math:`\theta(x)` of the x-data.
        Additionally, the amplitude is not assumed to follow a specific form.
        Techniques to compute a good initial guess for the polynomial coefficients inside
        a trigonometric function like this are not trivial. Instead, this analysis extracts the
        raw phase and runs fits on the extracted data to a polynomial :math:`\theta(x)` directly.

        The measured P1 values for a Ramsey X and Y experiment can be written in the form of
        a trignometric function taking the phase polynomial :math:`\theta(x)`:

        .. math::

            P_X =  \text{amp}(x) \cdot \cos \theta(x) + \text{offset},\\
            P_Y =  \text{amp}(x) \cdot \sin \theta(x) + \text{offset}.

        Hence the phase polynomial can be extracted as follows

        .. math::

            \theta(x) = \tan^{-1} \frac{P_Y}{P_X}.

        Because the arctangent is implemented by the ``atan2`` function
        defined in :math:`[-\pi, \pi]`, the computed :math:`\theta(x)` is unwrapped to
        ensure continuous phase evolution.

        We call attention to the fact that :math:`\text{amp}(x)` is also Stark tone amplitude
        dependent because of the qubit frequency dependence of the dephasing rate.
        In general :math:`\text{amp}(x)` is unpredictable due to dephasing from
        two-level systems distributed randomly in frequency
        or potentially due to qubit heating. This prevents us from precisely fitting
        the raw :math:`P_X`, :math:`P_Y` data. Fitting only the phase data makes the
        analysis robust to amplitude dependent dephasing.

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

            \theta^\nu(x) = c_1^\nu x + c_2^\nu x^2 + c_3^\nu x^3 + f_{\rm err},

        where :math:`\nu \in \{+, -\}`.
        The Stark shift is asymmetric with respect to :math:`x=0`, because of the
        anti-crossings of higher energy levels. In a typical transmon qubit,
        these levels appear only in :math:`f_S < 0` because of the negative anharmonicity.
        To precisely fit the results, this analysis uses different model parameters
        for positive (:math:`x > 0`) and negative (:math:`x < 0`) shift domains.

    # section: fit_parameters

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
            init_guess: 1e6.
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
            init_guess: -1e6.
            bounds: [-inf, 0]

        defpar c_3^-:
            desc: The cubic term coefficient of the negative Stark shift
                (fit parameter: ``stark_neg_coef_o3``).
            init_guess: 0.
            bounds: None

        defpar f_{\rm err}:
            desc: Constant phase accumulation which is independent of the Stark tone amplitude.
                (fit parameter: ``stark_ferr``).
            init_guess: 0
            bounds: None

    # section: see_also

        :class:`qiskit_experiments.library.characterization.analysis.ramsey_xy_analysis.RamseyXYAnalysis`

    """

    def __init__(self):

        models = [
            lmfit.models.ExpressionModel(
                expr="c1_pos * x + c2_pos * x**2 + c3_pos * x**3 + f_err",
                name="FREQpos",
            ),
            lmfit.models.ExpressionModel(
                expr="c1_neg * x + c2_neg * x**2 + c3_neg * x**3 + f_err",
                name="FREQneg",
            ),
        ]
        super().__init__(models=models)

    @classmethod
    def _default_options(cls):
        """Default analysis options.

        Analysis Options:
            pulse_len (float): Duration of effective Stark pulse in units of sec.
        """
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
            data_subfit_map={
                "Xpos": {"series": "X", "direction": "pos"},
                "Ypos": {"series": "Y", "direction": "pos"},
                "Xneg": {"series": "X", "direction": "neg"},
                "Yneg": {"series": "Y", "direction": "neg"},
            },
            plotter=ramsey_plotter,
            fit_category="freq",
            pulse_len=None,
        )

        return options

    def _freq_phase_coef(self) -> float:
        """Return a coefficient to convert frequency into phase value."""
        try:
            return 2 * np.pi * self.options.pulse_len
        except TypeError as ex:
            raise TypeError(
                "A float-value duration in units of sec of the Stark pulse must be provided. "
                f"The pulse_len option value {self.options.pulse_len} is not valid."
            ) from ex

    def _format_data(
        self,
        curve_data: curve.ScatterTable,
        category: str = "freq",
    ) -> curve.ScatterTable:

        curve_data = super()._format_data(curve_data, category="ramsey_xy")
        ramsey_xy = curve_data[curve_data.category == "ramsey_xy"]

        # Create phase data by arctan(Y/X)
        columns = list(curve_data.columns)
        phase_data = np.empty((0, len(columns)))
        y_mean = ramsey_xy.yval.mean()

        grouped = ramsey_xy.groupby("name")
        for m_id, direction in enumerate(("pos", "neg")):
            x_quadrature = grouped.get_group(f"X{direction}")
            y_quadrature = grouped.get_group(f"Y{direction}")
            if not np.array_equal(x_quadrature.xval, y_quadrature.xval):
                raise ValueError(
                    "Amplitude values of X and Y quadrature are different. "
                    "Same values must be used."
                )
            x_uarray = unp.uarray(x_quadrature.yval, x_quadrature.yerr)
            y_uarray = unp.uarray(y_quadrature.yval, y_quadrature.yerr)

            amplitudes = x_quadrature.xval.to_numpy()

            # pylint: disable=no-member
            phase = unp.arctan2(y_uarray - y_mean, x_uarray - y_mean)
            phase_n = unp.nominal_values(phase)
            phase_s = unp.std_devs(phase)

            # Unwrap phase
            # We assume a smooth slope and correct 2pi phase jump to minimize the change of the slope.
            unwrapped_phase = np.unwrap(phase_n)
            if amplitudes[0] < 0:
                # Preserve phase value closest to 0 amplitude
                unwrapped_phase = unwrapped_phase + (phase_n[-1] - unwrapped_phase[-1])

            # Store new data
            tmp = np.empty((len(amplitudes), len(columns)), dtype=object)
            tmp[:, columns.index("xval")] = amplitudes
            tmp[:, columns.index("yval")] = unwrapped_phase / self._freq_phase_coef()
            tmp[:, columns.index("yerr")] = phase_s / self._freq_phase_coef()
            tmp[:, columns.index("name")] = f"FREQ{direction}"
            tmp[:, columns.index("class_id")] = m_id
            tmp[:, columns.index("shots")] = x_quadrature.shots + y_quadrature.shots
            tmp[:, columns.index("category")] = category
            phase_data = np.r_[phase_data, tmp]

        return curve_data.append_list_values(other=phase_data)

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
        user_opt.p0.set_if_empty(
            c1_pos=0, c2_pos=1e6, c3_pos=0, c1_neg=0, c2_neg=-1e6, c3_neg=0, f_err=0
        )
        return user_opt

    def _create_analysis_results(
        self,
        fit_data: curve.CurveFitResult,
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        outcomes = super()._create_analysis_results(fit_data, quality, **metadata)

        # Combine fit coefficients
        coeffs = StarkCoefficients(
            pos_coef_o1=fit_data.ufloat_params["c1_pos"].nominal_value,
            pos_coef_o2=fit_data.ufloat_params["c2_pos"].nominal_value,
            pos_coef_o3=fit_data.ufloat_params["c3_pos"].nominal_value,
            neg_coef_o1=fit_data.ufloat_params["c1_neg"].nominal_value,
            neg_coef_o2=fit_data.ufloat_params["c2_neg"].nominal_value,
            neg_coef_o3=fit_data.ufloat_params["c3_neg"].nominal_value,
            offset=fit_data.ufloat_params["f_err"].nominal_value,
        )
        outcomes.append(
            AnalysisResultData(
                name="stark_coefficients",
                value=coeffs,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra=metadata,
            )
        )
        return outcomes

    def _create_figures(
        self,
        curve_data: curve.ScatterTable,
    ) -> List["matplotlib.figure.Figure"]:

        # plot unwrapped phase on first axis
        for d in ("pos", "neg"):
            sub_data = curve_data[(curve_data.name == f"FREQ{d}") & (curve_data.category == "freq")]
            self.plotter.set_series_data(
                series_name=f"F{d}",
                x_formatted=sub_data.xval.to_numpy(),
                y_formatted=sub_data.yval.to_numpy(),
                y_formatted_err=sub_data.yerr.to_numpy(),
            )

        # plot raw RamseyXY plot on second axis
        for name in ("Xpos", "Ypos", "Xneg", "Yneg"):
            sub_data = curve_data[(curve_data.name == name) & (curve_data.category == "ramsey_xy")]
            self.plotter.set_series_data(
                series_name=name,
                x_formatted=sub_data.xval.to_numpy(),
                y_formatted=sub_data.yval.to_numpy(),
                y_formatted_err=sub_data.yerr.to_numpy(),
            )

        # find base and amplitude guess
        ramsey_xy = curve_data[curve_data.category == "ramsey_xy"]
        offset_guess = 0.5 * (ramsey_xy.yval.min() + ramsey_xy.yval.max())
        amp_guess = 0.5 * np.ptp(ramsey_xy.yval)

        # plot frequency and Ramsey fit lines
        line_data = curve_data[curve_data.category == "fitted"]
        for direction in ("pos", "neg"):
            sub_data = line_data[line_data.name == f"FREQ{direction}"]
            if len(sub_data) == 0:
                continue
            xval = sub_data.xval.to_numpy()
            yn = sub_data.yval.to_numpy()
            ys = sub_data.yerr.to_numpy()
            yval = unp.uarray(yn, ys) * self._freq_phase_coef()

            # Ramsey fit lines are predicted from the phase fit line.
            # Note that this line doesn't need to match with the expeirment data
            # because Ramsey P1 data may fluctuate due to phase damping.

            # pylint: disable=no-member
            ramsey_cos = amp_guess * unp.cos(yval) + offset_guess
            ramsey_sin = amp_guess * unp.sin(yval) + offset_guess

            self.plotter.set_series_data(
                series_name=f"F{direction}",
                x_interp=xval,
                y_interp=yn,
            )
            self.plotter.set_series_data(
                series_name=f"X{direction}",
                x_interp=xval,
                y_interp=unp.nominal_values(ramsey_cos),
            )
            self.plotter.set_series_data(
                series_name=f"Y{direction}",
                x_interp=xval,
                y_interp=unp.nominal_values(ramsey_sin),
            )

            if np.isfinite(ys).all():
                self.plotter.set_series_data(
                    series_name=f"F{direction}",
                    y_interp_err=ys,
                )
                self.plotter.set_series_data(
                    series_name=f"X{direction}",
                    y_interp_err=unp.std_devs(ramsey_cos),
                )
                self.plotter.set_series_data(
                    series_name=f"Y{direction}",
                    y_interp_err=unp.std_devs(ramsey_sin),
                )
        return [self.plotter.figure()]

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        super()._initialize(experiment_data)

        # Set scaling factor to convert phase to frequency
        if "stark_length" in experiment_data.metadata:
            self.set_options(pulse_len=experiment_data.metadata["stark_length"])


class StarkP1SpectAnalysis(BaseAnalysis):
    """Analysis class for StarkP1Spectroscopy.

    # section: overview

        The P1 landscape is hardly predictable because of the random appearance of
        lossy TLS notches, and hence this analysis doesn't provide any
        generic mathematical model to fit the measurement data.
        A developer may subclass this to conduct own analysis.

        This analysis just visualizes the measured P1 values against Stark tone amplitudes.
        The tone amplitudes can be converted into the amount of Stark shift
        when the calibrated coefficients are provided in the analysis option,
        or the calibration experiment results are available in the result database.

    # section: see_also
        :class:`qiskit_experiments.library.driven_freq_tuning.StarkRamseyXYAmpScan`

    """

    @property
    def plotter(self) -> vis.CurvePlotter:
        """Curve plotter instance."""
        return self.options.plotter

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options.

        Analysis Options:
            plotter (Plotter): Plotter to visualize P1 landscape.
            data_processor (DataProcessor): Data processor to compute P1 value.
            stark_coefficients (Union[Dict, str]): Dictionary of Stark shift coefficients to
                convert tone amplitudes into amount of Stark shift. This dictionary must include
                all keys defined in :attr:`.StarkP1SpectAnalysis.stark_coefficients_names`,
                which are calibrated with :class:`.StarkRamseyXYAmpScan`.
                Alternatively, it searches for these coefficients in the result database
                when "latest" is set. This requires having the experiment service set in
                the experiment data to analyze.
            x_key (str): Key of the circuit metadata to represent x value.
        """
        options = super()._default_options()

        p1spect_plotter = vis.CurvePlotter(vis.MplDrawer())
        p1spect_plotter.set_figure_options(
            xlabel="Stark amplitude",
            ylabel="P(1)",
            xscale="quadratic",
        )

        options.update_options(
            plotter=p1spect_plotter,
            data_processor=dp.DataProcessor("counts", [dp.Probability("1")]),
            stark_coefficients=None,
            x_key="xval",
        )
        options.set_validator("stark_coefficients", StarkCoefficients)

        return options

    # pylint: disable=unused-argument
    def _run_spect_analysis(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        ydata_err: np.ndarray,
    ) -> list[AnalysisResultData]:
        """Run further analysis on the spectroscopy data.

        .. note::
            A subclass can overwrite this method to conduct analysis.

        Args:
            xdata: X values. This is either amplitudes or frequencies.
            ydata: Y values. This is P1 values measured at different Stark tones.
            ydata_err: Sampling error of the Y values.

        Returns:
            A list of analysis results.
        """
        return []

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:

        x_key = self.options.x_key

        # Get calibrated Stark tone coefficients
        if self.options.stark_coefficients is None and experiment_data.service is not None:
            # Get value from service
            stark_coeffs = retrieve_coefficients_from_service(
                service=experiment_data.service,
                backend_name=experiment_data.backend_name,
                qubit=experiment_data.metadata["physical_qubits"][0],
            )
        else:
            stark_coeffs = self.options.stark_coefficients

        # Compute P1 value and sampling error
        data = experiment_data.data()
        try:
            xdata = np.asarray([datum["metadata"][x_key] for datum in data], dtype=float)
        except KeyError as ex:
            raise DataProcessorError(
                f"X value key {x_key} is not defined in circuit metadata."
            ) from ex
        ydata_ufloat = self.options.data_processor(data)
        ydata = unp.nominal_values(ydata_ufloat)
        ydata_err = unp.std_devs(ydata_ufloat)

        # Convert x-axis of amplitudes into Stark shift by consuming calibrated parameters.
        if isinstance(stark_coeffs, StarkCoefficients):
            xdata = convert_amp_to_freq(
                amps=xdata,
                coeffs=stark_coeffs,
            )
            self.plotter.set_figure_options(
                xlabel="Stark shift",
                xval_unit="Hz",
                xscale="linear",
            )

        # Draw figures and create analysis results.
        self.plotter.set_series_data(
            series_name="stark_p1",
            x_formatted=xdata,
            y_formatted=ydata,
            y_formatted_err=ydata_err,
            x_interp=xdata,
            y_interp=ydata,
        )
        analysis_results = self._run_spect_analysis(xdata, ydata, ydata_err)

        return analysis_results, [self.plotter.figure()]
