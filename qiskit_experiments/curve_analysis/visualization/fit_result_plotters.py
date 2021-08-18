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
A systematic plotting function that draws full analysis result.

Note that plotter is a class that only has a class method to draw the image.
This is just like a function, but allows serialization via Enum.
"""

from typing import List, Tuple, Dict, Optional

import numpy as np
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter

from qiskit_experiments.curve_analysis.curve_data import SeriesDef, FitData, CurveData
from qiskit_experiments.framework import AnalysisResultData, FitVal
from qiskit_experiments.matplotlib import requires_matplotlib


try:
    from qiskit.utils import detach_prefix
except ImportError:

    # TODO remove this after Qiskit-terra #6885 becomes available
    def detach_prefix(value: float, decimal: Optional[int] = None) -> Tuple[float, str]:
        """A placeholder function. This will be imported from qiskit terra."""
        prefactors = {
            -15: "f",
            -12: "p",
            -9: "n",
            -6: "µ",
            -3: "m",
            0: "",
            3: "k",
            6: "M",
            9: "G",
            12: "T",
            15: "P",
        }

        if not np.isreal(value):
            raise ValueError(f"Input should be real number. Cannot convert {value}.")

        if np.abs(value) != 0:
            pow10 = int(np.floor(np.log10(np.abs(value)) / 3) * 3)
        else:
            pow10 = 0

        if pow10 > 0:
            mant = value / pow(10, pow10)
        else:
            mant = value * pow(10, -pow10)

        if decimal is not None:
            mant = np.round(mant, decimal)
            if mant >= 1000:
                mant /= 1000
                pow10 += 3

        if pow10 not in prefactors:
            raise ValueError(f"Value is out of range: {value}")

        return mant, prefactors[pow10]


from .style import PlotterStyle
from .curves import plot_scatter, plot_errorbar, plot_curve_fit


class MplDrawSingleCanvas:
    """A plotter to draw a single canvas figure for fit result."""

    @classmethod
    @requires_matplotlib
    def draw(
        cls,
        curves: List[Tuple[SeriesDef, CurveData, CurveData]],
        tick_labels: Dict[str, str],
        fit_data: FitData,
        result_entries: List[AnalysisResultData],
        style: Optional[PlotterStyle] = None,
        axis: Optional["matplotlib.axes.Axes"] = None,
    ) -> "pyplot.Figure":
        """Create a fit result of all curves in the single canvas.

        Args:
            curves: A tuple of series definition with a set of curve data
                representing raw data points and formatted data points.
            tick_labels: Dictionary of axis label information. Axis units and label for x and y
                value should be explained.
            fit_data: fit data generated by the analysis.
            result_entries: List of analysis result data entries. If
                :py:class:`~qiskit_experiments.database_service.db_fitval.FitVal` object is a value,
                that entry will be shown in the fit report.
            style: Optional. A configuration object to modify the appearance of the figure.
            axis: Optional. A matplotlib Axis object.

        Returns:
            A matplotlib figure of the curve fit result.
        """
        if axis is None:
            figure = pyplot.figure(figsize=style.figsize)
            axis = figure.subplots(nrows=1, ncols=1)
        else:
            figure = axis.get_figure()

        for series_def, raw_data, format_data in curves:
            # plot raw data if data is formatted
            if not np.array_equal(raw_data.y, format_data.y):
                plot_scatter(xdata=raw_data.x, ydata=raw_data.y, ax=axis, zorder=0)

            # plot formatted data
            if np.all(np.isnan(format_data.y_err)):
                sigma = None
            else:
                sigma = np.nan_to_num(format_data.y_err)
            plot_errorbar(
                xdata=format_data.x,
                ydata=format_data.y,
                sigma=sigma,
                ax=axis,
                label=series_def.name,
                marker=series_def.plot_symbol,
                color=series_def.plot_color,
                zorder=1,
                linestyle="",
            )

            # plot fit curve
            if fit_data:
                plot_curve_fit(
                    func=series_def.fit_func,
                    result=fit_data,
                    ax=axis,
                    color=series_def.plot_color,
                    zorder=2,
                    fit_uncertainty=series_def.plot_fit_uncertainty,
                )

        # add legend
        if len(curves) > 1:
            axis.legend(loc=style.legend_loc)

        # get axis scaling factor
        for this_axis in ("x", "y"):
            sub_axis = getattr(axis, this_axis + "axis")
            unit = tick_labels[this_axis + "val_unit"]
            label = tick_labels[this_axis + "label"]
            if unit:
                maxv = np.max(np.abs(sub_axis.get_data_interval()))
                scaled_maxv, prefix = detach_prefix(maxv, decimal=3)
                prefactor = scaled_maxv / maxv
                # pylint: disable=cell-var-from-loop
                sub_axis.set_major_formatter(FuncFormatter(lambda x, p: f"{x * prefactor: g}"))
                sub_axis.set_label_text(f"{label} [{prefix}{unit}]", fontsize=style.axis_label_size)
            else:
                sub_axis.set_label_text(label, fontsize=style.axis_label_size)
                axis.ticklabel_format(axis=this_axis, style="sci", scilimits=(-3, 3))

        # write analysis report
        if fit_data:
            report_str = write_fit_report(result_entries)
            report_str += r"Fit $\chi^2$ = " + f"{fit_data.reduced_chisq: .4f}"

            report_handler = axis.text(
                *style.fit_report_rpos,
                report_str,
                ha="center",
                va="top",
                size=style.fit_report_text_size,
                transform=axis.transAxes,
            )

            bbox_props = dict(boxstyle="square, pad=0.3", fc="white", ec="black", lw=1, alpha=0.8)
            report_handler.set_bbox(bbox_props)

        axis.tick_params(labelsize=style.tick_label_size)
        axis.grid(True)

        return figure


def write_fit_report(result_entries: List[AnalysisResultData]) -> str:
    """A function that generates fit reports documentation from list of data.

    Args:
        result_entries: List of data entries.

    Returns:
        Documentation of fit reports.
    """
    analysis_description = ""

    def format_val(float_val: float) -> str:
        if np.abs(float_val) < 1e-3 or np.abs(float_val) > 1e3:
            return f"{float_val: .4e}"
        return f"{float_val: .4f}"

    for res in result_entries:
        if isinstance(res.value, FitVal) and not res.name.startswith("@Parameters_"):
            fitval = res.value
            if fitval.unit:
                # unit is defined. do detaching prefix, i.e. 1000 Hz -> 1 kHz
                val, val_prefix = detach_prefix(fitval.value, decimal=3)
                val_unit = val_prefix + fitval.unit
                value_repr = f"{val: .3f}"
                if fitval.stderr is not None:
                    # with stderr
                    err, err_prefix = detach_prefix(fitval.stderr, decimal=3)
                    err_unit = err_prefix + fitval.unit
                    if val_unit == err_unit:
                        # same value scaling, same prefix
                        value_repr += f" \u00B1 {err: .2f} {val_unit}"
                    else:
                        # different value scaling, different prefix
                        value_repr += f" {val_unit} \u00B1 {err: .2f} {err_unit}"
                else:
                    # without stderr, just append unit
                    value_repr += f" {val_unit}"
            else:
                # unit is not defined. raw value formatting is performed.
                value_repr = format_val(fitval.value)
                if fitval.stderr is not None:
                    # with stderr
                    value_repr += f" \u00B1 {format_val(fitval.stderr)}"

            analysis_description += f"{res.name} = {value_repr}\n"

    return analysis_description
