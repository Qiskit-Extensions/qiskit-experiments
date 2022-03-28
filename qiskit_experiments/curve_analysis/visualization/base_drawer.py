# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Curve drawer abstract class."""

from abc import ABC, abstractmethod
from typing import List, Dict, Sequence, Union, Optional, Callable

from qiskit_experiments.curve_analysis.curve_data import FitData
from qiskit_experiments.framework import Options, AnalysisResultData


class BaseCurveDrawer(ABC):
    """Abstract class for the serializable Qiskit Experiments curve drawer.

    A subclass must implement following abstract methods.

    initialize_canvas

        This method should implement a protocol to initialize a drawing canvas
        with user input ``axis`` object. Note that curve analysis drawer
        supports visualization in the 2D inset axes. This method
        should first check the drawing options if axis object is provided,
        and initialize the axis only when it is not provided.
        Once axis is initialized, this is set to the instance member ``self._axis``.

    format_canvas

        This method should implement a protocol to format the appearance of canvas.
        Typically, it updates axis and tick labels. Note that the axis SI unit
        may be specified in the drawing options. In this case, axis numbers should be
        auto-scaled with the unit prefix.

    draw_raw_data

        This method is called after data processing is completed.
        This method draws raw experiment data points on the canvas.

    draw_formatted_data

        This method is called after data formatting is completed.
        The formatted data might be averaged over the same x values,
        or smoothed by the filtering algorithms, depending on how analysis class is implemented.
        This method is called with error bars of y values and the name of the curve.

    draw_fit_lines

        This method is called after fitting is completed and when there is valid fit outcome.
        This method is called with the fitting model that can generate y values with
        error bars. Interpolated x value should be internally generated.

    draw_fit_report

        This method is called after fitting is completed and when there is valid fit outcome.
        This method is called with the list of analysis results and the reduced chi-squared values.
        The fit report should be generated to show this information on the canvas.

    """

    def __init__(self):
        self._options = self._default_options()
        self._set_options = set()
        self._axis = None

    @property
    def options(self) -> Options:
        """Return the drawing options."""
        return self._options

    @classmethod
    def _default_options(cls) -> Options:
        """Return default draw options.

        Draw Options:
            axis (Any): Arbitrary object that can be used as a drawing canvas.
            subplots (Tuple[int, int]): Number of rows and columns when the experimental
                result is drawn in the multiple windows.
            xlabel (str): X-axis label string of the output figure.
            ylabel (str): Y-axis label string of the output figure.
            xlim (Tuple[float, float]): Min and max value of the horizontal axis.
                If not provided, it is automatically scaled based on the input data points.
            ylim (Tuple[float, float]): Min and max value of the vertical axis.
                If not provided, it is automatically scaled based on the input data points.
            xval_unit (str): SI unit of x values. No prefix is needed here.
                For example, when the x values represent time, this option will be just "s"
                rather than "ms". In the output figure, the prefix is automatically selected
                based on the maximum value in this axis. If your x values are in [1e-3, 1e-4],
                they are displayed as [1 ms, 10 ms]. This option is likely provided by the
                analysis class rather than end-users. However, users can still override
                if they need different unit notation. By default, this option is set to ``None``,
                and no scaling is applied. If nothing is provided, the axis numbers will be
                displayed in the scientific notation.
            yval_unit (str): Unit of y values. See ``xval_unit`` for details.
            figsize (Tuple[int, int]): A tuple of two numbers representing the size of
                the output figure (width, height). Note that this is applicable
                only when ``axis`` object is not provided. If any canvas object is provided,
                the figure size associated with the axis is preferentially applied.
            legend_loc (str): Vertical and horizontal location of the curve legend window in
                a single string separated by a space. This defaults to ``center right``.
                Vertical position can be ``upper``, ``center``, ``lower``.
                Horizontal position can be ``right``, ``center``, ``left``.
            tick_label_size (int): Size of text representing the axis tick numbers.
            axis_label_size (int): Size of text representing the axis label.
            fit_report_rpos (Tuple[int, int]): A tuple of numbers showing the location of
                the fit report window. These numbers are horizontal and vertical position
                of the top left corner of the window in the relative coordinate
                on the output figure, i.e. ``[0, 1]``.
                The fit report window shows the selected fit parameters and the reduced
                chi-squared value.
            fit_report_text_size (int): Size of text in the fit report window.
            plot_sigma (List[Tuple[float, float]]): A list of two number tuples
                showing the configuration to write confidence intervals for the fit curve.
                The first argument is the relative sigma (n_sigma), and the second argument is
                the transparency of the interval plot in ``[0, 1]``.
                Multiple n_sigma intervals can be drawn for the single curve.
        """
        return Options(
            axis=None,
            subplots=(1, 1),
            xlabel=None,
            ylabel=None,
            xlim=None,
            ylim=None,
            xval_unit=None,
            yval_unit=None,
            figsize=(8, 5),
            legend_loc="center right",
            tick_label_size=14,
            axis_label_size=16,
            fit_report_rpos=(0.6, 0.95),
            fit_report_text_size=14,
            plot_sigma=[(1.0, 0.7), (3.0, 0.3)],
        )

    def set_options(self, **fields):
        """Set the drawing options.
        Args:
            fields: The fields to update the options
        """
        self._options.update_options(**fields)
        self._set_options = self._set_options.union(fields)

    @abstractmethod
    def initialize_canvas(self):
        """Initialize the drawing canvas."""

    @abstractmethod
    def format_canvas(self):
        """Final cleanup for the canvas appearance."""

    @abstractmethod
    def draw_raw_data(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        ax_index: Optional[int] = None,
        **options,
    ):
        """Draw raw data.

        Args:
            x_data: X values.
            y_data: Y values.
            ax_index: Index of canvas if multiple inset axis exist.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def draw_formatted_data(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        y_err_data: Sequence[float],
        name: Optional[str] = None,
        ax_index: Optional[int] = None,
        **options,
    ):
        """Draw formatted data that used for fitting.

        Args:
            x_data: X values.
            y_data: Y values.
            y_err_data: Standard deviation of Y values.
            name: Name of this curve.
            ax_index: Index of canvas if multiple inset axis exist.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def draw_fit_lines(
        self,
        fit_function: Callable,
        signature: List[str],
        fit_result: FitData,
        fixed_params: Dict[str, float],
        ax_index: Optional[int] = None,
        **options,
    ):
        """Draw fit lines.

        Args:
            fit_function: The function defines a single curve.
            signature: The fit parameters associated with the function.
            fit_result: The result of fit.
            fixed_params: The parameter fixed during the fitting.
            ax_index: Index of canvas if multiple inset axis exist.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def draw_fit_report(
        self,
        analysis_results: List[AnalysisResultData],
        chisq: Union[float, Dict[str, float]],
    ):
        """Draw text box that shows fit reports.

        Args:
            analysis_results: List of analysis result entries containing fit parameters.
            chisq: Chi-squared value from the fitting. If this is provided as a dictionary,
                the key is also shown with the chi-squared value.
        """

    @property
    @abstractmethod
    def figure(self):
        """Return figure object handler of the canvas object.

        Note that figure and axis might be different plot when a user provide
        an axis object which is a part of other multiple axis figure.
        This method returns the entire figure object, which is saved in the database.
        """

    def __json_encode__(self):
        return {
            "cls": type(self),
            "options": self._set_options,
        }

    @classmethod
    def __json_decode__(cls, value):
        instance = cls()
        if "options" in value:
            instance.set_options(**value["options"])
        return instance
