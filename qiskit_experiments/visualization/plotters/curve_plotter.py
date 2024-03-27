# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Plotter for curve fits, specifically from :class:`.CurveAnalysis`."""
from typing import List

import numpy as np
from uncertainties import UFloat

from qiskit_experiments.curve_analysis.utils import analysis_result_to_repr
from qiskit_experiments.framework import Options
from .base_plotter import BasePlotter


class CurvePlotter(BasePlotter):
    """A plotter class to plot results from :class:`.CurveAnalysis`.

    ``CurvePlotter`` plots results from curve fits, which includes

        - Raw results as a scatter plot.
        - Processed results with standard deviations/confidence intervals.
        - Interpolated fit results from the curve analysis.
        - Confidence interval for the fit results.
        - A report on the performance of the fit.
    """

    @classmethod
    def expected_series_data_keys(cls) -> List[str]:
        """Returns the expected series data keys supported by this plotter.

        Data Keys:
            x: X-values for raw results.
            y: Y-values for raw results. Goes with ``x``.
            x_formatted: X-values for processed results.
            y_formatted: Y-values for processed results. Goes with ``x_formatted``.
            y_formatted_err: Error in ``y_formatted``, to be plotted as error-bars.
            x_interp: Interpolated X-values for a curve fit.
            y_interp: Y-values corresponding to the fit for ``y_interp`` X-values.
            y_interp_err: The standard deviations of the fit for each X-value in
                ``y_interp``. This data key relates to the option ``plot_sigma``.
            x_residuals: The X-values for the residual plot.
            y_residuals: The residual from the fitting.
        """
        return [
            "x",
            "y",
            "x_formatted",
            "y_formatted",
            "y_formatted_err",
            "x_interp",
            "y_interp",
            "y_interp_err",
            "x_residuals",
            "y_residuals",
        ]

    @classmethod
    def expected_supplementary_data_keys(cls) -> List[str]:
        """Returns the expected figures data keys supported by this plotter.

        This plotter generates a single text box, i.e. fit report, by digesting the
        provided supplementary data. The style and position of the report is controlled
        by ``textbox_rel_pos`` and ``textbox_text_size`` style parameters in
        :class:`PlotStyle`.

        Data Keys:
            primary_results: A list of :class:`.AnalysisResultData` objects to be shown
                in the fit report window. Typically, these are fit parameter values or
                secondary quantities computed from multiple fit parameters.
            fit_red_chi: The best reduced-chi squared value of the fit curves. If the
                fit consists of multiple sub-fits, this will be a dictionary keyed on
                the analysis name. Otherwise, this is a single float value of a
                particular analysis.
        """
        return [
            "primary_results",
            "fit_red_chi",
        ]

    @classmethod
    def _default_options(cls) -> Options:
        """Return curve-plotter specific default plotter options.

        Options:
            plot_sigma (List[Tuple[float, float]]): A list of two number tuples showing
                the configuration to write confidence intervals for the fit curve. The
                first argument is the relative sigma (n_sigma), and the second argument
                is the transparency of the interval plot in ``[0, 1]``. Multiple n_sigma
                intervals can be drawn for the same curve.

        """
        options = super()._default_options()
        options.plot_sigma = [(1.0, 0.7), (3.0, 0.3)]
        return options

    @classmethod
    def _default_figure_options(cls) -> Options:
        r"""Return curve-plotter specific default figure options.

        Figure Options:
            report_red_chi2_label (str): The label for the reduced-chi squared entry of
                the fit report. Defaults to the Python string literal
                ``"reduced-$\\chi^2$"``, corresponding to the formatted string
                reduced-:math:`\chi^2`.
        """
        fig_opts = super()._default_figure_options()
        fig_opts.report_red_chi2_label = "reduced-$\\chi^2$"

        return fig_opts

    def _plot_figure(self):
        """Plots a curve fit figure."""
        for ser in self.series:
            # Scatter plot with error-bars
            plotted_formatted_data = False
            if self.data_exists_for(ser, ["x_formatted", "y_formatted", "y_formatted_err"]):
                x, y, yerr = self.data_for(ser, ["x_formatted", "y_formatted", "y_formatted_err"])
                if x is not None and y is not None:
                    self.drawer.scatter(x, y, y_err=yerr, name=ser, zorder=2, legend=True)
                    plotted_formatted_data = True

            # Scatter plot
            if self.data_exists_for(ser, ["x", "y"]):
                x, y = self.data_for(ser, ["x", "y"])
                if x is not None and y is not None:
                    options = {
                        "zorder": 1,
                    }
                    # If we plotted formatted data, differentiate scatter points
                    # by setting normal X-Y markers to gray.
                    if plotted_formatted_data:
                        options["color"] = "gray"
                    # If we didn't plot formatted data, the X-Y markers should be used for the legend.
                    # We add it to ``options`` so it's easier to pass to ``scatter``.
                    if not plotted_formatted_data:
                        options["legend"] = True
                    self.drawer.scatter(x, y, name=ser, **options)

            # Line and confidence interval plot for fit
            if self.data_exists_for(ser, ["x_interp", "y_interp"]):
                x, y = self.data_for(ser, ["x_interp", "y_interp"])
                if x is not None and y is not None:
                    self.drawer.line(x, y, name=ser, zorder=3)
                    if self.data_exists_for(ser, ["y_interp_err"]):
                        if (y_err := self.data_for(ser, ["y_interp_err"])[0]) is not None:
                            y_err = np.array(y_err, dtype=float)
                            for n_sigma, alpha in self.options.plot_sigma:
                                self.drawer.filled_y_area(
                                    x,
                                    y + n_sigma * y_err,
                                    y - n_sigma * y_err,
                                    name=ser,
                                    alpha=alpha,
                                    zorder=5,
                                )

            # Plot residuals
            if self.data_exists_for(ser, ["x_residuals", "y_residuals"]):
                # check if we cancel residuals plotting
                if self.options.get("style", {}).get("style_name") != "canceled_residuals":
                    series_name = ser + "_residuals"
                    x, y = self.data_for(ser, ["x_residuals", "y_residuals"])
                    self.drawer.scatter(x, y, name=series_name, legend=True)

            # Fit report
            report = self._write_report()
            if len(report) > 0:
                self.drawer.textbox(report)

    def _write_report(self) -> str:
        """Write fit report with supplementary_data.

        Subclass can override this method to customize fit report. By default, this
        writes important fit parameters and chi-squared value of the fit in the fit
        report. The ``report_red_chi2_label`` figure option controls the label for the
        chi-squared entries in the report.

        Returns:
            Fit report.
        """
        report = ""

        if "primary_results" in self.supplementary_data:
            lines = []
            for outcome in self.supplementary_data["primary_results"]:
                if isinstance(outcome.value, (float, UFloat)):
                    lines.append(analysis_result_to_repr(outcome))
            report += "\n".join(lines)

        if "fit_red_chi" in self.supplementary_data:
            red_chi = self.supplementary_data["fit_red_chi"]
            if len(report) > 0:
                report += "\n"
            if isinstance(red_chi, float):
                report += f"{self.figure_options.report_red_chi2_label} = {red_chi: .4g}"
            else:
                # Composite curve analysis reporting multiple chi-sq values.
                # This is usually given by a dict keyed on fit group name.

                # Add gap between primary-results and reduced-chi squared as
                # we have multiple values to display. This is easier to read.
                if len(report) > 0:
                    report += "\n"

                # Created indented text of reduced-chi squared results.
                report += f"{self.figure_options.report_red_chi2_label} per fit\n"
                lines = []
                for mod_name, mod_chi in red_chi.items():
                    lines.append(f"  * {mod_name}: {mod_chi: .4g}")
                report += "\n".join(lines)

        return report
