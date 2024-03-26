# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Drawer abstract class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from qiskit_experiments.framework import Options

from qiskit_experiments.framework.configs import DrawerConfig

from ..style import PlotStyle
from ..utils import ExtentTuple

SeriesName = Union[str, int, float]


class BaseDrawer(ABC):
    """Abstract class for the serializable Qiskit Experiments figure drawer.

    # section: overview

    A drawer may be implemented by different drawer backends such as matplotlib or
    Plotly. Sub-classes that wrap these backends by subclassing :class:`BaseDrawer` must
    implement the following abstract methods.

    .. describe:: initialize_canvas

        This method should implement a protocol to initialize a drawer canvas with user
        input ``axis`` object. Note that ``drawer`` supports visualization of experiment
        results in multiple canvases tiled into N (row) x M (column) inset grids, which
        is specified in the option ``subplots``. By default, this is N=1, M=1 and thus
        no inset grid will be initialized. The data points to draw might be provided
        with a canvas number defined in :attr:`SeriesDef.canvas` which defaults to
        ``None``, i.e. no-inset grids.

        This method should first check the drawer options (:attr:`options`) for the axis
        object and initialize the axis only when it is not provided by the options. Once
        axis is initialized, this is set to the instance member ``self._axis``.

    .. describe:: format_canvas

        This method formats the appearance of the canvas. Typically, it updates axis and
        tick labels. Note that the axis SI unit may be specified in the drawer
        figure_options. In this case, axis numbers should be auto-scaled with the unit
        prefix.

    .. rubric:: Drawing Methods

    .. describe:: scatter

        This method draws scatter points on the canvas, like a scatter-plot, with
        optional error-bars in both the X and Y axes.

    .. describe:: line

        This method plots a line from provided X and Y values.

    .. describe:: filled_y_area

        This method plots a shaped region bounded by upper and lower Y-values. This
        method is typically called with interpolated x and a pair of y values that
        represent the upper and lower bound within certain confidence interval. If this
        is called multiple times, it may be necessary to set the transparency so that
        overlapping regions can be distinguished.

    .. describe:: filled_x_area

        This method plots a shaped region bounded by upper and lower X-values, as a
        function of Y-values. This method is a rotated analogue of
        :meth:`filled_y_area`.

    .. describe:: textbox

        This method draws a text-box on the canvas, which is a rectangular region
        containing some text.

    .. rubric:: Legends

    Legends are generated based off of drawn graphics and their labels or names. These
    are managed by individual drawer subclasses, and generated when the
    :meth:`format_canvas` method is called. Legend entries are created when any drawing
    function is called with ``legend=True``. There are three parameters in drawing
    functions that are relevant to legend generation: ``name``, ``label``, and
    ``legend``. If a user would like the graphics drawn onto a canvas to be used as the
    graphical component of a legend entry; they should set ``legend=True``. The legend
    entry label can be defined in three locations: the ``label`` parameter of drawing
    functions, the ``"label"`` entry in ``series_params``, and the ``name`` parameter of
    drawing functions. These three possible label variables have a search hierarchy
    given by the order in the aforementioned list. If one of the label variables is
    ``None``, the next is used. If all are ``None``, a legend entry is not generated for
    the given series.

    The recommended way to customize the legend entries is as follows:

    1. Set the labels in the ``series_params`` option, keyed on the series names.
    2. Initialize the canvas.
    3. Call relevant drawing methods to create the figure. When calling the drawing
       method that creates the graphic you would like to use in the legend, set
       ``legend=True``. For example, ``drawer.scatter(...,legend=True)`` would use
       the scatter points as the legend graphics for the given series.
    4. Format the canvas and call :meth:`figure` to get the figure.

    .. rubric:: Options and Figure Options

    Drawers have both :attr:`options` and :attr:`figure_options` available to set
    parameters that define how to draw and what is drawn, respectively.
    :class:`BasePlotter` is similar in that it also has ``options`` and
    ``figure_options``. The former contains class-specific variables that define how an
    instance behaves. The latter contains figure-specific variables that typically
    contain values that are drawn on the canvas, such as text. For details on the
    difference between the two sets of options, see the documentation for
    :class:`BasePlotter`.

    .. note::
        If a drawer instance is used with a plotter, then there is the potential for
        any figure option to be overwritten with their value from the plotter. This
        means that the drawer instance would be modified indirectly when the
        :meth:`BasePlotter.figure` method is called. This must be kept in mind when
        creating subclasses of :class:`BaseDrawer`.

    """

    def __init__(self):
        """Create a BaseDrawer instance."""
        # Normal options. Which includes the drawer axis, subplots, and default style.
        self._options = self._default_options()
        # A set of changed options for serialization.
        self._set_options = set()

        # Figure options which are typically updated by a plotter instance. Figure options include the
        # axis labels, figure title, and a custom style instance.
        self._figure_options = self._default_figure_options()
        # A set of changed figure options for serialization.
        self._set_figure_options = set()

        # The initialized axis/axes, set by `initialize_canvas`.
        self._axis = None

    @property
    def options(self) -> Options:
        """Return the drawer options."""
        return self._options

    @property
    def figure_options(self) -> Options:
        """Return the figure options.

        These are typically updated by a plotter instance, and thus may change. It is
        recommended to set figure options in a parent :class:`BasePlotter` instance that
        contains the :class:`BaseDrawer` instance.
        """
        return self._figure_options

    @classmethod
    def _default_options(cls) -> Options:
        """Return default drawer options.

        Options:
            axis (Any): Arbitrary object that can be used as a canvas.
            subplots (Tuple[int, int]): Number of rows and columns when the experimental
                result is drawn in the multiple windows.
            default_style (PlotStyle): The default style for drawer. This must contain
                all required style parameters for :class:`drawer`, as is defined in
                :meth:`PlotStyle.default_style()`. Subclasses can add extra required
                style parameters by overriding :meth:`_default_style`.
        """
        return Options(
            axis=None,
            subplots=(1, 1),
            default_style=cls._default_style(),
        )

    @classmethod
    def _default_style(cls) -> PlotStyle:
        return PlotStyle.default_style()

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
            custom_style (PlotStyle): The style definition to use when drawing. This
                overwrites style parameters in ``default_style`` in :attr:`options`.
                Defaults to an empty PlotStyle instance (i.e., ``PlotStyle()``).
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
            custom_style=PlotStyle(),
        )
        options.set_validator("xscale", ["linear", "log", "symlog", "logit", "quadratic", None])
        options.set_validator("yscale", ["linear", "log", "symlog", "logit", "quadratic", None])

        return options

    def set_options(self, **fields):
        """Set the drawer options.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If an unknown options is encountered.
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
            fields: The fields to update the figure options

        Raises:
            AttributeError: If an unknown figure option is encountered.
        """
        for field in fields:
            if not hasattr(self._figure_options, field):
                raise AttributeError(
                    f"Figure options field {field} is not valid for {type(self).__name__}"
                )
        self._figure_options.update_options(**fields)
        self._set_figure_options = self._set_figure_options.union(fields)

    @property
    def style(self) -> PlotStyle:
        """The combined plot style for this drawer.

        The returned style instance is a combination of :attr:`options.default_style`
        and :attr:`figure_options.custom_style`. Style parameters set in
        ``custom_style`` override those set in ``default_style``. If ``custom_style`` is
        not an instance of :class:`PlotStyle`, the returned style is equivalent to
        ``default_style``.

        Returns:
            The plot style for this drawer.
        """
        if isinstance(self.figure_options.custom_style, PlotStyle):
            return PlotStyle.merge(self.options.default_style, self.figure_options.custom_style)
        return self.options.default_style

    @abstractmethod
    def initialize_canvas(self):
        """Initialize the drawer canvas."""

    @abstractmethod
    def format_canvas(self):
        """Final cleanup for the canvas appearance."""

    def label_for(self, name: Optional[SeriesName], label: Optional[SeriesName]) -> Optional[str]:
        """Get the legend label for the given series, with optional overrides.

        This method determines the legend label for a series, with optional overrides
        ``label`` and the ``"label"`` entry in the ``series_params`` option (see
        :attr:`options`). ``label`` is returned if it is not ``None``, as this is the
        override with the highest priority. If it is ``None``, then the drawer will look
        for a ``"label"`` entry in ``series_params``, for the series identified by
        ``name``. If this entry doesn't exist, or is ``None``, then ``name`` is used as
        the label. If all these options are ``None``, then ``None`` is returned;
        signifying that a legend entry for the provided series should not be generated.
        Note that :meth:`label_for` will convert ``name`` to ``str`` when it is
        returned.

        Args:
            name: The name of the series.
            label: Optional label override.

        Returns:
            The legend entry label, or ``None``.
        """
        if label is not None:
            return str(label)

        if name is not None:
            return self.figure_options.series_params.get(name, {}).get("label", str(name))
        return None

    @abstractmethod
    def scatter(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        x_err: Optional[Sequence[float]] = None,
        y_err: Optional[Sequence[float]] = None,
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        """Draw scatter points, with optional error-bars.

        Args:
            x_data: X values.
            y_data: Y values.
            x_err: Optional error for X values.
            y_err: Optional error for Y values.
            name: Name of this series.
            label: Optional legend label to override ``name`` and ``series_params``.
            legend: Whether the drawn area must have a legend entry. Defaults to False.
                The series label in the legend will be ``label`` if it is not None. If
                it is, then ``series_params`` is searched for a ``"label"`` entry for
                the series identified by ``name``. If this is also ``None``, then
                ``name`` is used as the fallback. If no ``name`` is provided, then no
                legend entry is generated.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def line(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        """Draw a line.

        Args:
            x_data: X values.
            y_data: Y values.
            name: Name of this series.
            label: Optional legend label to override ``name`` and ``series_params``.
            legend: Whether the drawn area must have a legend entry. Defaults to False.
                The series label in the legend will be ``label`` if it is not None. If
                it is, then ``series_params`` is searched for a ``"label"`` entry for
                the series identified by ``name``. If this is also ``None``, then
                ``name`` is used as the fallback. If no ``name`` is provided, then no
                legend entry is generated.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def hline(
        self,
        y_value: float,
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        """Draw a horizontal line.

        Args:
            y_value: Y value for line.
            name: Name of this series.
            label: Optional legend label to override ``name`` and ``series_params``.
            legend: Whether the drawn area must have a legend entry. Defaults to False.
                The series label in the legend will be ``label`` if it is not None. If
                it is, then ``series_params`` is searched for a ``"label"`` entry for
                the series identified by ``name``. If this is also ``None``, then
                ``name`` is used as the fallback. If no ``name`` is provided, then no
                legend entry is generated.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def filled_y_area(
        self,
        x_data: Sequence[float],
        y_ub: Sequence[float],
        y_lb: Sequence[float],
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        """Draw filled area as a function of x-values.

        Args:
            x_data: X values.
            y_ub: The upper boundary of Y values.
            y_lb: The lower boundary of Y values.
            name: Name of this series.
            label: Optional legend label to override ``name`` and ``series_params``.
            legend: Whether the drawn area must have a legend entry. Defaults to False.
                The series label in the legend will be ``label`` if it is not None. If
                it is, then ``series_params`` is searched for a ``"label"`` entry for
                the series identified by ``name``. If this is also ``None``, then
                ``name`` is used as the fallback. If no ``name`` is provided, then no
                legend entry is generated.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def filled_x_area(
        self,
        x_ub: Sequence[float],
        x_lb: Sequence[float],
        y_data: Sequence[float],
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        """Draw filled area as a function of y-values.

        Args:
            x_ub: The upper boundary of X values.
            x_lb: The lower boundary of X values.
            y_data: Y values.
            name: Name of this series.
            label: Optional legend label to override ``name`` and ``series_params``.
            legend: Whether the drawn area must have a legend entry. Defaults to False.
                The series label in the legend will be ``label`` if it is not None. If
                it is, then ``series_params`` is searched for a ``"label"`` entry for
                the series identified by ``name``. If this is also ``None``, then
                ``name`` is used as the fallback. If no ``name`` is provided, then no
                legend entry is generated.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def textbox(
        self,
        description: str,
        rel_pos: Optional[Tuple[float, float]] = None,
        **options,
    ):
        """Draw text box.

        Args:
            description: A string to be drawn inside a report box.
            rel_pos: Relative position of the text-box. If None, the default
                ``textbox_rel_pos`` from the style is used.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def image(
        self,
        data: np.ndarray,
        extent: Optional[ExtentTuple] = None,
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        cmap: Optional[Union[str, Any]] = None,
        cmap_use_series_colors: bool = False,
        colorbar: bool = False,
        **options,
    ):
        """Draw an image of numerical values, series names, or RGB/A values.

        Args:
            data: The two-/three-dimensional data defining an image. If
                ``data.dims==2``, then the pixel colors are determined by ``cmap`` and
                ``cmap_use_series_colors``. If ``data.dims==3``, then it is assumed that
                ``data`` contains either RGB or RGBA data; which requires the third
                dimension to have length ``3`` or ``4`` respectively. For RGB/A data,
                the elements of ``data`` must be floats or integers in the range 0-1 and
                0-255 respectively. If the data is two-dimensional, there is no limit on
                the range of the values if they are numerical. If
                ``cmap_use_series_colors=True``, then ``data`` contains series names;
                which can be strings or numerical values, as long as they are
                appropriate series names.
            extent: An optional tuple ``(x_min, x_max, y_min, y_max)`` which defines a
                rectangular region within which the values inside ``data`` should be
                plotted. The units of ``extent`` are the same as those of the X and Y
                axes for the axis. If None, the extent of the image is taken as ``(0,
                data.shape[0], 0, data.shape[1])``. Default is None.
            name: Name of this image. Used to lookup ``canvas`` and ``label`` in
                ``series_params``.
            label: An optional label for the colorbar, if ``colorbar=True``.
            cmap: Optional colormap for assigning colors to the image values, if
                ``data`` is not an RGB/A image. ``cmap`` must be a string or object
                instance which is recognized by the drawer. Defaults to None.
            cmap_use_series_colors: Whether to assign colors to the image based on
                series colors, where the values inside ``data`` are series names. If
                ``cmap_use_series_colors=True``,``cmap`` is ignored. This only works for
                two-dimensional images as three-dimensional ``data`` contains explicit
                colors as RGB/A values. If ``len(data.shape)=3``,
                ``cmap_use_series_colours`` is ignored. Defaults to False.
            colorbar: Whether to add a bar showing the color-value mapping for the
                image. Defaults to False.
            options: Valid options for the drawer backend API.
        """

    @property
    @abstractmethod
    def figure(self):
        """Return figure object handler to be saved in the database."""

    def config(self) -> DrawerConfig:
        """Return the config dictionary for this drawer."""
        options = dict((key, getattr(self._options, key)) for key in self._set_options)
        figure_options = dict(
            (key, getattr(self._figure_options, key)) for key in self._set_figure_options
        )
        return DrawerConfig(
            cls=type(self),
            options=options,
            figure_options=figure_options,
        )

    @classmethod
    def from_config(cls, config: Union[DrawerConfig, Dict]) -> "BaseDrawer":
        """Initialize a drawer class from analysis config"""
        if isinstance(config, dict):
            config = DrawerConfig(**config)
        # Create Drawer instance
        ret = cls()
        if config.options:
            ret.set_options(**config.options)
        if config.figure_options:
            ret.set_figure_options(**config.figure_options)
        return ret

    def __json_encode__(self):
        return self.config()

    @classmethod
    def __json_decode__(cls, value):
        return cls.from_config(value)
