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
"""Base plotter abstract class"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from qiskit_experiments.framework import Options
from qiskit_experiments.visualization.drawers import BaseDrawer, SeriesName

from ..style import PlotStyle


class BasePlotter(ABC):
    """An abstract class for the serializable figure plotters.

    # section: overview

    A plotter takes data from an experiment analysis class or experiment and plots a
    given figure using a drawing backend. Sub-classes define the kind of figure created
    and the expected data.

    Data is split into series and supplementary data. Series data is grouped by series
    name (``Union[str, int, float]``). For :class:`CurveAnalysis`, this is the model
    name for a curve fit. For series data associated with a single series name and
    supplementary data, data values are identified by a data key (str). Different data
    per series and figure must have a different data key to avoid overwriting values.
    Experiment and analysis results can be passed to the plotter so appropriate graphics
    can be drawn on the figure canvas. Series data is added to the plotter using
    :meth:`set_series_data` whereas supplementary data is added using
    :meth:`set_supplementary_data`. Series and supplementary data are retrieved using
    :meth:`data_for` and :attr:`supplementary_data` respectively.

    Series data contains values to be plotted on a canvas, such that the data can be
    grouped into subsets identified by their series name. Series names can be thought of
    as legend labels for the plotted data, and as curve names for a curve fit.
    Supplementary data is not associated with a series or curve and is instead only
    associated with the figure. Examples include analysis reports or other text that is
    drawn onto the figure canvas.

    .. rubric:: Options and Figure Options

    Plotters have both :attr:`options` and :attr:`figure_options` available to set
    parameters that define how to plot and what is plotted. :class:`BaseDrawer` is
    similar in that it also has ``options`` and ``figure_options``. The former contains
    class-specific variables that define how an instance behaves. The latter contains
    figure-specific variables that typically contain values that are drawn on the
    canvas, such as text.

    For example, :class:`BasePlotter` has an ``axis`` option that can be set to the
    canvas on which the figure should be drawn. This changes how the plotter works in
    that it changes where the figure is drawn. :class:`BasePlotter` has an ``xlabel``
    figure option that can be set to change the text drawn next to the X-axis in the
    final figure. As the value of this option will be drawn on the figure, it is a
    figure option.

    As plotters need a drawer to generate a figure, and the drawer needs to know what to
    draw, figure options are passed to :attr:`drawer` when the :meth:`figure` method is
    called. Any figure options that are defined in both the plotters
    :attr:`figure_options` attribute and the drawers ``figure_options`` attribute are
    copied to the drawer: i.e., :meth:`BaseDrawer.set_figure_options` is called for each
    common figure option, setting the value of the option to the value stored in the
    plotter.

    .. note::
        If a figure option called "foo" is not set in the drawer's figure options
        (:attr:`BaseDrawer.figure_options`) but is set in the plotter's figure options
        (:attr:`figure_options`), it will not be copied over to the drawer when the
        :meth:`figure` method is called. This means that some figure options from the
        plotter may be unused by the drawer. :class:`BasePlotter` and its subclasses
        filter these options before setting them in the drawer, as subclasses of
        :class:`BaseDrawer` may add additional figure options. To make validation
        easier and the code cleaner, the :meth:`figure` method conducts this check
        before setting figure options in the drawer.

    .. rubric:: Example

    .. code-block:: python

        plotter = MyPlotter(MyDrawer())

        # MyDrawer contains the following figure_options with default values.
        plotter.drawer.figure_options.xlabel
        plotter.drawer.figure_options.ylabel

        # MyDrawer does NOT contain the following figure option
        # plotter.drawer.figure_options.unknown_variable    # Raises an error as it
                                                            # does not exist in
                                                            # `drawer.figure_options`.

        # If we set the following figure options, they will be set in the drawer as
        # they are defined in `plotter.drawer.figure_options`.
        plotter.set_figure_options(xlabel="Frequency", ylabel="Fidelity")

        # During a call to `plotter.figure()`, the drawer's figure options are updated.
        # The following values would then be returned from the drawer.
        plotter.drawer.figure_options.xlabel                # returns "Frequency"
        plotter.drawer.figure_options.ylabel                # returns "Fidelity"

        # If we set the following option and figure option, they will NOT be set in the
        # drawer as the drawer doesn't contain default values for these option names.
        plotter.set_options(plot_fit=False)                 # Example plotter option
        plotter.set_figure_options(unknown_variable=5e9)    # Example figure option

        # As `plot_fit` is not a figure option, it is not set in the drawer.
        plotter.drawer.options.plot_fit     # Would raise an error if no default
                                            # exists, or return a different value to
                                            # `plotter.options.plot_fit`.

        # As `unknown_variable` is not set in the drawer's figure options, it is not set
        # during a # call to the `figure()` method.
        # plotter.drawer.figure_options.unknown_variable    # Raises an error as it
                                                            # does not exist in
                                                            # `drawer.figure_options`.

    Attributes:
        drawer (BaseDrawer): The drawer to use when plotting.
    """

    def __init__(self, drawer: BaseDrawer):
        """Create a new plotter instance.

        Args:
            drawer: The drawer to use when creating the figure.
        """
        # Data to be plotted, such as scatter points, interpolated fits, and confidence intervals
        self._series_data: Dict[SeriesName, Dict[str, Any]] = {}
        # Data that isn't directly associated with a single series, such as text or fit reports.
        self._supplementary_data: Dict[str, Any] = {}

        # Options for the plotter
        self._options = self._default_options()
        # Plotter options that have changed, for serialization.
        self._set_options = set()

        # Figure options that are updated in the drawer when `plotter.figure()` is called
        self._figure_options = self._default_figure_options()
        # Figure options that have changed, for serialization.
        self._set_figure_options = set()

        self.drawer: BaseDrawer = drawer

    @property
    def supplementary_data(self) -> Dict[str, Any]:
        """Additional data for the figure being plotted, that isn't associated with a
        series.

        Supplementary data includes text, fit reports, or other data that is associated
        with the figure but not an individual series. It is typically data additional to
        the direct results of an experiment.
        """
        return self._supplementary_data

    @property
    def series_data(self) -> Dict[SeriesName, Dict[str, Any]]:
        """Data for series being plotted.

        Series data includes data such as scatter points, interpolated fit values, and
        standard deviations. Series data is grouped by series name (``Union[str, int,
        float]``) and then by a data key (``str``). Though series data can be accessed
        through :attr:`series_data`, it is recommended to access them with
        :meth:`data_for` and :meth:`data_exists_for` as they allow for easier access to
        nested values and can handle multiple data keys in one query.

        Returns:
            A dictionary containing series data.
        """
        return self._series_data

    @property
    def series(self) -> List[SeriesName]:
        """Series names that have been added to this plotter."""
        return list(self._series_data.keys())

    def data_keys_for(self, series_name: SeriesName) -> List[str]:
        """Returns a list of data keys for the given series.

        Args:
            series_name: The series name for which to return the data keys, i.e., the
                types of data for each series.

        Returns:
            The list of data keys for data in the plotter associated with the given
            series. If the series has not been added to the plotter, an empty list is
            returned.
        """
        return list(self._series_data.get(series_name, []))

    def data_for(
        self, series_name: SeriesName, data_keys: Union[str, List[str]]
    ) -> Tuple[Optional[Any], ...]:
        """Returns data associated with the given series.

        The returned tuple contains the data, associated with ``data_keys``, in the same
        orders as they are provided. For example,

        .. code-block:: python

            plotter.set_series_data("seriesA", x=data.x, y=data.y, yerr=data.yerr)

            # The following calls are equivalent.
            x, y, yerr = plotter.data_for("seriesA", ["x", "y", "yerr"])
            x, y, yerr = data.x, data.y, data.yerr

            # Retrieving a single data key returns a tuple. Note the comma after ``x``.
            x, = plotter.data_for("seriesA", "x")

        :meth:`data_for` is intended to be used by sub-classes of :class:`BasePlotter`
        when plotting in the :meth:`_plot_figure` method.

        Args:
            series_name: The series name for the given series.
            data_keys: List of data keys for the data to be returned. If a single
                data key is given as a string, it is wrapped in a list.

        Returns:
            A tuple of data associated with the given series, identified by
            ``data_keys``. If no data has been set for a data key, None is returned for
            the associated tuple entry.
        """

        # We may be given a single data key, but we need a list for the rest of the function.
        if not isinstance(data_keys, list):
            data_keys = [data_keys]

        # The series doesn't exist in the plotter data, return None for each data key in the output.
        if series_name not in self._series_data:
            return (None,) * len(data_keys)

        return tuple(self._series_data[series_name].get(key, None) for key in data_keys)

    def set_series_data(self, series_name: SeriesName, **data_kwargs):
        """Sets data for the given series.

        Note that if data has already been assigned for the given series and data key,
        it will be overwritten with the new values. ``set_series_data`` will warn if the
        data key is unexpected; i.e., not within those returned by
        :meth:`expected_series_data_keys`.

        Args:
            series_name: The name of the given series.
            data_kwargs: The data to be added, where the keyword is the data key.
        """
        # Warn if the data keys are not expected.
        unknown_data_keys = [
            data_key for data_key in data_kwargs if data_key not in self.expected_series_data_keys()
        ]
        for unknown_data_key in unknown_data_keys:
            warnings.warn(
                f"{self.__class__.__name__} encountered an unknown data key {unknown_data_key}. It may "
                "not be used by the plotter class."
            )

        # Set data
        if series_name not in self._series_data:
            self._series_data[series_name] = {}
        self._series_data[series_name].update(**data_kwargs)

    def clear_series_data(self, series_name: Optional[SeriesName] = None):
        """Clear series data for this plotter.

        Args:
            series_name: The series name identifying which data should be cleared. If
                None, all series data is cleared. Defaults to None.
        """
        if series_name is None:
            self._series_data = {}
        elif series_name in self._series_data:
            self._series_data.pop(series_name)

    def set_supplementary_data(self, **data_kwargs):
        """Sets supplementary data for the plotter.

        Supplementary data differs from series data in that it is not associate with a
        series name. Fit reports are examples of supplementary data as they contain fit
        results from an analysis class, such as the "goodness" of a curve fit.

        Note that if data has already been assigned for the given data key, it will be
        overwritten with the new values. ``set_supplementary_data`` will warn if the
        data key is unexpected; i.e., not within those returned by
        :meth:`expected_supplementary_data_keys`.

        """

        # Warn if any data keys are not expected.
        unknown_data_keys = [
            data_key
            for data_key in data_kwargs
            if data_key not in self.expected_supplementary_data_keys()
        ]
        for unknown_data_key in unknown_data_keys:
            warnings.warn(
                f"{self.__class__.__name__} encountered an unknown data key {unknown_data_key}. It may "
                "not be used by the plotter class."
            )

        self._supplementary_data.update(**data_kwargs)

    def clear_supplementary_data(self):
        """Clears supplementary data."""
        self._supplementary_data = {}

    def data_exists_for(self, series_name: SeriesName, data_keys: Union[str, List[str]]) -> bool:
        """Returns whether the given data keys exist for the given series.

        Args:
            series_name: The name of the given series.
            data_keys: The data keys to be checked.

        Returns:
            True if all data keys have values assigned for the given series. False if at
            least one does not have a value assigned.
        """
        if not isinstance(data_keys, list):
            data_keys = [data_keys]

        # Handle non-existent series name
        if series_name not in self._series_data:
            return False

        return all(key in self._series_data[series_name] for key in data_keys)

    @abstractmethod
    def _plot_figure(self):
        """Generates a figure using :attr:`drawer` and :meth:`data`.

        Sub-classes must override this function to plot data using the drawer. This
        function is called by :meth:`figure` when :attr:`drawer` can be used to draw on
        the canvas.
        """

    def figure(self) -> Any:
        """Generates and returns a figure for the already provided series and
        supplementary data.

        :meth:`figure` calls :meth:`_plot_figure`, which is overridden by sub-classes.
        Before and after calling :meth:`_plot_figure`;
        :meth:`~BaseDrawer._configure_drawer`, :meth:`~BaseDrawer.initialize_canvas` and
        :meth:`~BaseDrawer.format_canvas` are called on the drawer respectively.

        Returns:
            A figure generated by :attr:`drawer`, of the same type as ``drawer.figure``.
        """
        # Initialize drawer, to copy axis, subplots, style, and figure options across.
        self._configure_drawer()

        # Initialize canvas, which creates subplots, assigns axis labels, etc.
        self.drawer.initialize_canvas()

        # Plot figure for given subclass. This is the core of BasePlotter subclasses.
        self._plot_figure()

        # Final formatting of canvas, which sets axis limits etc.
        self.drawer.format_canvas()

        # Return whatever figure is created by the drawer.
        return self.drawer.figure

    @classmethod
    @abstractmethod
    def expected_series_data_keys(cls) -> List[str]:
        """Returns the expected series data keys supported by this plotter."""

    @classmethod
    @abstractmethod
    def expected_supplementary_data_keys(cls) -> List[str]:
        """Returns the expected supplementary data keys supported by this plotter."""

    @property
    def options(self) -> Options:
        """Options for the plotter.

        Options for a plotter modify how the class generates a figure. This includes an
        optional axis object, being the drawer canvas. Make sure verify whether the
        option you want to set is in :attr:`options` or :attr:`figure_options`.
        """
        return self._options

    @property
    def figure_options(self) -> Options:
        """Figure options for the plotter and its drawer.

        Figure options differ from normal options (:attr:`options`) in that the plotter
        passes figure options on to the drawer when creating a figure (when
        :meth:`figure` is called). This way :attr:`drawer` can draw an appropriate
        figure. An example of a figure option is the x-axis label.
        """
        return self._figure_options

    @classmethod
    def _default_options(cls) -> Options:
        """Return default plotter options.

        Options:
            axis (Any): Arbitrary object that can be used as a drawing canvas.
            subplots (Tuple[int, int]): Number of rows and columns when the experimental
                result is drawn in the multiple windows.
            style (PlotStyle): The style definition to use when plotting.
                This overwrites figure option `custom_style` set in :attr:`drawer`. The
                default is an empty style object, and such the default :attr:`drawer`
                plotting style will be used.
        """
        return Options(
            axis=None,
            subplots=(1, 1),
            style=PlotStyle(),
        )

    @classmethod
    def _default_figure_options(cls) -> Options:
        """Return default figure options.

        Figure Options:
            xlabel (Union[str, List[str]]): X-axis label string of the output figure. If
                there are multiple columns in the canvas, this could be a list of labels.
            ylabel (Union[str, List[str]]): Y-axis label string of the output figure. If
                there are multiple rows in the canvas, this could be a list of labels.
            xlim (Union[Tuple[float, float], List[Tuple[float, float]]): Min and max value
                of the horizontal axis. If not provided, it is automatically scaled based
                on the input data points. If there are multiple columns in the canvas,
                this could be a list of xlims.
            ylim (Union[Tuple[float, float], List[Tuple[float, float]]): Min and max value
                of the vertical axis. If not provided, it is automatically scaled based
                on the input data points. If there are multiple rows in the canvas,
                this could be a list of ylims.
            xval_unit (Union[str, List[str]]): Unit of x values.
                No scaling prefix is needed here as this is controlled by ``xval_unit_scale``.
                If there are multiple columns in the canvas, this could be a list of xval_units.
            yval_unit (Union[str, List[str]]): Unit of y values.
                No scaling prefix is needed here as this is controlled by ``yval_unit_scale``.
                If there are multiple rows in the canvas, this could be a list of yval_units.
            xval_unit_scale (Union[bool, List[bool]]): Whether to add an SI unit prefix to
                ``xval_unit`` if needed. For example, when the x values represent time and
                ``xval_unit="s"``, ``xval_unit_scale=True`` adds an SI unit prefix to
                ``"s"`` based on X values of plotted data. In the output figure, the
                prefix is automatically selected based on the maximum value in this
                axis. If your x values are in [1e-3, 1e-4], they are displayed as [1 ms,
                10 ms]. By default, this option is set to ``True``. If ``False`` is
                provided, the axis numbers will be displayed in the scientific notation.
                If there are multiple columns in the canvas, this could be a list of xval_unit_scale.
            yval_unit_scale (Union[bool, List[bool]]): Whether to add an SI unit prefix to
                ``yval_unit`` if needed. See ``xval_unit_scale`` for details.
                If there are multiple rows in the canvas, this could be a list of yval_unit_scale.
            xscale (str): The scaling of the x-axis, such as ``log`` or ``linear``.
            yscale (str): The scaling of the y-axis, such as ``log`` or ``linear``.
            figure_title (str): Title of the figure. Defaults to None, i.e. nothing is
                shown.
            sharex (bool): Set True to share x-axis ticks among sub-plots.
            sharey (bool): Set True to share y-axis ticks among sub-plots.
            series_params (Dict[str, Dict[str, Any]]): A dictionary of parameters for
                each series. This is keyed on the name for each series. Sub-dictionary
                is expected to have the following three configurations, "canvas",
                "color", "symbol" and "label"; "canvas" is the integer index of axis
                (when multi-canvas plot is set), "color" is the color of the drawn
                graphics, "symbol" is the series marker style for scatter plots, and
                "label" is a user provided series label that appears in the legend.
        """
        options = Options(
            xlabel=None,
            ylabel=None,
            xlim=None,
            ylim=None,
            xval_unit=None,
            yval_unit=None,
            xval_unit_scale=True,
            yval_unit_scale=True,
            xscale=None,
            yscale=None,
            sharex=True,
            sharey=True,
            figure_title=None,
            series_params={},
        )
        options.set_validator("xscale", ["linear", "log", "symlog", "logit", "quadratic", None])
        options.set_validator("yscale", ["linear", "log", "symlog", "logit", "quadratic", None])

        return options

    def set_options(self, **fields):
        """Set the plotter options.

        Args:
            fields: The fields to update in options.

        Raises:
            AttributeError: If an unknown option is encountered.
        """
        for field in fields:
            if not hasattr(self._options, field):
                raise AttributeError(
                    f"Options field {field} is not valid for {type(self).__name__}"
                )
        self._options.update_options(**fields)
        self._set_options = self._set_options.union(fields)

    def set_figure_options(self, **fields):
        """Set the figure options.

        Args:
            fields: The fields to update in figure options.
        """
        # Don't check if any option in fields already exists (like with `set_options`), as figure options
        # are passed to `.drawer` which may have other figure options. Any figure option that isn't set
        # in `.drawer.figure_options` won't be set anyway. Setting `.drawer.figure_options` only occurs
        # in `.figure()`, so we can't compare to `.drawer.figure_options` now as `.drawer` may be changed
        # between now and the call to `.figure()`.
        self._figure_options.update_options(**fields)
        self._set_figure_options = self._set_figure_options.union(fields)

    def _configure_drawer(self):
        """Configures :attr:`drawer` before plotting.

        The following actions are taken:

        1. ``axis``, ``subplots``, and ``style`` are passed to :attr:`drawer`.
        2. ``figure_options`` in :attr:`drawer` are updated based on values set in
           the plotter :attr:`figure_options`

        These steps are different as all figure options could be passed to
        :attr:`drawer`, if the drawer already has a figure option with the same name.
        ``axis``, ``subplots``, and ``style`` are the only plotter options (from
        :attr:`options`) passed to :attr:`drawer` in :meth:`_configure_drawer`. This is
        done as these options make more sense as an option for a plotter, given the
        interface of :class:`BasePlotter`.
        """
        ## Axis, subplots, and style
        if self.options.axis:
            self.drawer.set_options(axis=self.options.axis)
        if self.options.subplots:
            self.drawer.set_options(subplots=self.options.subplots)
        self.drawer.set_figure_options(custom_style=self.options.style)

        # Convert options to dictionaries for easy comparison of all options/fields.
        _drawer_figure_options = self.drawer.figure_options.__dict__
        _plotter_figure_options = self.figure_options.__dict__

        # If an option exists in drawer.figure_options AND in self.figure_options, set the drawer's
        # figure option value to that from the plotter.
        for opt_key in _drawer_figure_options:
            if opt_key in _plotter_figure_options:
                _drawer_figure_options[opt_key] = _plotter_figure_options[opt_key]

        # Use drawer.set_figure_options so figure options are serialized.
        self.drawer.set_figure_options(**_drawer_figure_options)

    def config(self) -> Dict:
        """Return the config dictionary for this drawing."""
        options = dict((key, getattr(self._options, key)) for key in self._set_options)
        figure_options = dict(
            (key, getattr(self._figure_options, key)) for key in self._set_figure_options
        )
        drawer = self.drawer.__json_encode__()

        return {
            "cls": type(self),
            "options": options,
            "figure_options": figure_options,
            "drawer": drawer,
        }

    def __json_encode__(self):
        return self.config()

    @classmethod
    def __json_decode__(cls, value):
        ## Process drawer as it's needed to create a plotter
        drawer_values = value["drawer"]
        # We expect a subclass of BaseDrawer
        drawer_cls: BaseDrawer = drawer_values["cls"]
        drawer = drawer_cls.__json_decode__(drawer_values)

        # Create plotter instance
        instance = cls(drawer)
        if "options" in value:
            instance.set_options(**value["options"])
        if "figure_options" in value:
            instance.set_figure_options(**value["figure_options"])
        return instance
