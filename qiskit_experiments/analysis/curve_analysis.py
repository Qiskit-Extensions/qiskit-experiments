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
Analysis class for curve fitting.
"""
# pylint: disable=invalid-name

import dataclasses
import functools
import inspect
from typing import Any, Dict, List, Tuple, Callable, Union, Optional

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.providers.options import Options

from qiskit_experiments.analysis import plotting
from qiskit_experiments.analysis.curve_fitting import multi_curve_fit
from qiskit_experiments.analysis.data_processing import level2_probability
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.experiment_data import AnalysisResult, ExperimentData


@dataclasses.dataclass(frozen=True)
class SeriesDef:
    """Description of curve."""

    name: str
    fit_func: Callable
    filter_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    plot_color: str = "black"
    plot_symbol: str = "o"


class CurveAnalysis(BaseAnalysis):
    """A base class for curve fit type analysis.

    The subclasses can override class attributes to define the behavior of
    data extraction and fitting. This docstring describes how code developers can
    create a new curve fit analysis subclass inheriting from this base class.

    Class Attributes:

        __x_key__: Key in the circuit metadata under which to find the value for
            the horizontal axis.
        __series__: A set of data points that will be fit to a the same parameters
            in the fit function. If this analysis contains multiple curves,
            the same number of series definitions should be listed.
            Each series definition is SeriesDef element, that may be initialized with::

                name: Name of the curve. This is arbitrary data field, but should be unique.
                fit_func: Callback function to perform fit.
                filter_kwargs: Circuit metadata key and value associated with this curve.
                    The data points of the curve is extracted from ExperimentData based on
                    this information.
                plot_color: String color representation of this series in the plot.
                plot_symbol: String formatter of the scatter of this series in the plot.

            See the Examples below for more details.
        __fit_label_desc__: Dict of parameter names and its representation shows in the
            result figure as an analysis report.
        __plot_xlabel__: Label of x axis of result plot.
        __plot_ylabel__: Label of y axis of result plot.


    Examples:

        A fitting for single exponential decay curve
        ============================================

        In this type of experiment, the analysis deals with a single curve.
        Thus filter_kwargs is not necessary defined.
        In this example, fit value and error of the parameter ``lamb`` labeled by
        a Greek lambda symbol are written  in the figure.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __x_key__ = "scan_val"

                __series__ = [
                    SeriesDef(
                        name="my_experiment1",
                        fit_func=lambda x, p0, p1, p2:
                            exponential_decay(x, amp=p0, lamb=p1, baseline=p2),
                    ),
                ]

                __plot_labels__ = {"lamb": "\u03BB"}


        A fitting for two exponential decay curve with partly shared parameter
        ======================================================================

        In this type of experiment, the analysis deals with two curves.
        We need a __series__ definition for each curve, and filter_kwargs should be
        properly defined to separate each curve series.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __x_key__ = "scan_val"

                __series__ = [
                    SeriesDef(
                        name="my_experiment1",
                        fit_func=lambda x, p0, p1, p2, p3:
                            exponential_decay(x, amp=p0, lamb=p1, baseline=p3),
                        filter_kwargs={"experiment": 1},
                        plot_color="red",
                        plot_symbol="^",
                    ),
                    SeriesDef(
                        name="my_experiment2",
                        fit_func=lambda x, p0, p1, p2, p3:
                            exponential_decay(x, amp=p0, lamb=p2, baseline=p3),
                        filter_kwargs={"experiment": 2},
                        plot_color="blue",
                        plot_symbol="o",
                    ),
                ]

                __plot_labels__ = {"lamb": "\u03BB"}


        A fitting for two trigonometric curves with the same parameter
        =============================================================

        In this type of experiment, the analysis deals with two different curves.
        However the parameters are shared with both functions.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __x_key__ = "scan_val"

                __series__ = [
                    SeriesDef(
                        name="my_experiment1",
                        fit_func=lambda x, p0, p1, p2, p3:
                            cos(x, amp=p0, freq=p1, phase=p2, baseline=p3),
                        filter_kwargs={"experiment": 1},
                        plot_color="red",
                        plot_symbol="^",
                    ),
                    SeriesDef(
                        name="my_experiment2",
                        fit_func=lambda x, p0, p1, p2, p3:
                            sin(x, amp=p0, freq=p1, phase=p2, baseline=p3),
                        filter_kwargs={"experiment": 2},
                        plot_color="blue",
                        plot_symbol="o",
                    )
                ]

                __plot_labels__ = {"lamb": "\u03BB"}


    Notes:
        This CurveAnalysis class provides several private methods that subclasses can override.

        - Customize figure generation:
            Override :meth:`~self._create_figures`. For example, here you can create
            arbitrary number of new figures or upgrade the default figure appearance.

        - Customize pre-data processing:
            Override :meth:`~self._data_pre_processing`. For example, here you can
            take a mean over y values for the same x value, or apply smoothing to y values.

        - Customize post-analysis data processing:
            Override :meth:`~self._post_processing`. For example, here you can
            calculate new entity from fit values. Such as EPC of RB experiment.

        - Customize fitting options:
            Override :meth:`~self._setup_fitting`. For example, here you can
            calculate initial guess from experiment data and setup fitter options.

        - Customize data processor calibration:
            Override :meth:`~Self._calibrate_data_processor`. This is special subroutine
            that is only called when a DataProcessor instance is used as the data processor.
            You can take arbitrary data from experiment result and setup your processor.

        Note that other private methods are not expected to be overridden.
        If you forcibly override these methods, the behavior of analysis logic is not well tested
        and we cannot guarantee it works as expected (you may suffer from bugs).
        Instead, you can open an issue in qiskit-experiment github to upgrade this class
        with proper unittest framework.

        https://github.com/Qiskit/qiskit-experiments/issues
    """

    #: str: Metadata key representing a scanned value.
    __x_key__ = "xval"

    #: List[SeriesDef]: List of mapping representing a data series
    __series__ = None

    #: Dict[str, str]: Mapping of fit parameters and representation in the figure label.
    __fit_label_desc__ = None

    #: str: X axis label
    __plot_xlabel__ = "x value"

    #: str: Y axis label
    __plot_ylabel__ = "y value"

    @classmethod
    def _default_options(cls):
        """Return default data processing options.

        Options:
            plot: Set ``True`` to create figure for fit result.
            add_label: Set ``True`` to write fit report in the figure.
            ax: Optional. A matplotlib axis object to draw.
            base_fitter: A callback function to perform fitting with formatted data.
            data_processor: A callback function to format experiment data.
        """
        return Options(
            plot=True,
            add_label=True,
            ax=None,
            base_fitter=multi_curve_fit,
            data_processor=level2_probability,
        )

    def _create_figures(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_sigmas: np.ndarray,
        series: np.ndarray,
        analysis_results: AnalysisResult,
        axis: Optional["AxisSubplot"] = None,
        add_label: bool = True,
    ) -> List["Figure"]:
        """Create new figures with the fit result and raw data.

        Subclass can override this method to create different type of figures.

        Args:
            x_values: Full data set of x values.
            y_values: Full data set of y values.
            y_sigmas: Full data set of y sigmas.
            series: An integer array representing a mapping of data location to series index.
            analysis_results: Analysis result containing fit parameters.
            axis: User provided axis to draw result.
            add_label: Set ``True`` to add analysis result label.

        Returns:
            List of figures.
        """
        if plotting.HAS_MATPLOTLIB:

            if axis is None:
                figure = plotting.pyplot.figure()
                axis = figure.subplots(nrows=1, ncols=1)
            else:
                figure = axis.get_figure()

            axis.set_xlabel(self.__plot_xlabel__, fontsize=16)
            axis.set_ylabel(self.__plot_ylabel__, fontsize=16)

            for series_def in self.__series__:

                # plot raw data

                xdata, ydata, _ = self._subset_data(
                    series_def.name, x_values, y_values, y_sigmas, series
                )
                plotting.plot_scatter(xdata=xdata, ydata=ydata, ax=axis, zorder=0)

                # plot formatted data

                xdata, ydata, sigma = self._subset_data(
                    series_def.name, *self._pre_processing(x_values, y_values, y_sigmas, series)
                )
                plotting.plot_errorbar(
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    ax=axis,
                    label=series_def.name,
                    marker=series_def.plot_symbol,
                    color=series_def.plot_color,
                    zorder=1,
                )

                # plot fit curve

                if analysis_results["success"]:
                    plotting.plot_curve_fit(
                        func=series_def.fit_func,
                        result=analysis_results,
                        ax=axis,
                        color=series_def.plot_color,
                        zorder=2,
                    )

                # format axis

                if len(self.__series__) > 1:
                    axis.legend()
                axis.tick_params(labelsize=14)
                axis.grid(True)

            # write analysis report

            if add_label and analysis_results["success"]:
                # write fit status in the plot
                analysis_description = "Analysis Reports:\n"
                for par_name, label in self.__fit_label_desc__.items():
                    try:
                        # fit value
                        pind = analysis_results["popt_keys"].index(par_name)
                        pval = analysis_results["popt"][pind]
                        perr = analysis_results["popt_err"][pind]
                    except ValueError:
                        # maybe post processed value
                        pval = analysis_results[par_name]
                        perr = analysis_results[f"{par_name}_err"]
                    analysis_description += f"  \u25B7 {label} = {pval: .3e} \u00B1 {perr: .3e}\n"
                chisq = analysis_results["reduced_chisq"]
                analysis_description += f"Fit \u03C7-squared = {chisq: .4f}"

                axis.text(
                    axis.get_xlim()[0],
                    axis.get_ylim()[1],
                    analysis_description,
                    ha="left",
                    va="bottom",
                    size=12,
                )

            return [figure]
        else:
            return list()

    # pylint: disable = unused-argument
    def _setup_fitting(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_sigmas: np.ndarray,
        series: np.ndarray,
        **options,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """An analysis subroutine that is called to set fitter options.

        This subroutine takes full data array and user-input fit options.
        Subclasses can override this method to provide own fitter options
        such as initial guesses.

        Note that this subroutine can generate multiple fit options.
        If multiple options are provided, fitter runs multiple times for each fit option,
        and find the best result measured by the reduced chi-squared value.

        Args:
            x_values: Full data set of x values.
            y_values: Full data set of y values.
            y_sigmas: Full data set of y sigmas.
            series: An integer array representing a mapping of data location to series index.
            options: User provided fit options.

        Returns:
            List of FitOptions that are passed to fitter function.
        """
        return options

    def _pre_processing(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_sigmas: np.ndarray,
        series: np.ndarray,
        **options,
    ) -> Tuple[np.ndarray, ...]:
        """An optional subroutine to perform data pre-processing.

        Subclasses can override this method to apply pre-precessing to data values to fit.
        Otherwise the analysis uses extracted data values as-is.

        For example,

        - Take mean over all y data values with the same x data value
        - Apply smoothing to y values to deal with noisy observed values

        Args:
            x_values: Numpy float array to represent X values.
            y_values: Numpy float array to represent Y values.
            y_sigmas: Numpy float array to represent Y errors.
            series: Numpy integer array to represent mapping of data to series.
            options: Analysis options.

        Returns:
            Numpy array tuple of pre-processed (x_values, y_values, y_sigmas, series).
        """
        return x_values, y_values, y_sigmas, series

    def _post_processing(self, analysis_result: AnalysisResult, **options) -> AnalysisResult:
        """Calculate new quantity from the fit result.

        Subclasses can override this method to do post analysis.

        Args:
            analysis_result: Analysis result containing fit result.
            options: Analysis options.

        Returns:
            New AnalysisResult instance containing the result of post analysis.
        """
        return analysis_result

    def _extract_curves(
        self,
        experiment_data: ExperimentData,
        data_processor: Union[Callable, DataProcessor],
    ) -> Tuple[np.ndarray, ...]:
        """Extract curve data from experiment data.

        .. notes::
            The target metadata properties to define each curve entry is described by
            the class attribute __series__. This method returns the same numbers
            of curve data entries as one defined in this attribute.
            The returned CurveData entry contains circuit metadata fields that are
            common to the entire curve scan, i.e. series-level metadata.

        Args:
            experiment_data: ExperimentData object to fit parameters.
            data_processor: A callable or DataProcessor instance to format data into numpy array.
                This should take list of dictionary and returns two tuple of float values
                that represent a y value and an error of it.

        Returns:
            List of ``CurveEntry`` containing x-values, y-values, and y values sigma.

        Raises:
            QiskitError:
                - When __x_key__ is not defined in the circuit metadata.
        """

        def _is_target_series(datum, **filters):
            try:
                return all(datum["metadata"][key] == val for key, val in filters.items())
            except KeyError:
                return False

        # Extract X, Y, Y_sigma data
        data = experiment_data.data()

        try:
            x_values = [datum["metadata"][self.__x_key__] for datum in data]
        except KeyError as ex:
            raise QiskitError(
                f"X value key {self.__x_key__} is not defined in circuit metadata."
            ) from ex

        y_values, y_sigmas = zip(*map(data_processor, data))

        # Format data
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        y_sigmas = np.asarray(y_sigmas, dtype=float)

        # Find series (invalid data is labeled as -1)
        series = -1 * np.ones(x_values.size, dtype=int)
        for idx, series_def in enumerate(self.__series__):
            data_index = np.asarray(
                [_is_target_series(datum, **series_def.filter_kwargs) for datum in data], dtype=bool
            )
            series[data_index] = idx

        return x_values, y_values, y_sigmas, series

    def _format_fit_options(self, **fitter_options) -> Dict[str, Any]:
        """Format fitting option args to dictionary of parameter names.

        Args:
            fitter_options: Fit options generated by `self._setup_fitting`.

        Returns:
            Formatted fit options.

        Raises:
            QiskitError:
                - When fit functions have different signature.
            KeyError:
                - When fit option is dictionary but key doesn't match with parameter names.
                - When initial guesses are not provided.
            ValueError:
                - When fit option is array but length doesn't match with parameter number.
        """
        # check fit function signatures
        fsigs = set()
        for series_def in self.__series__:
            fsigs.add(inspect.signature(series_def.fit_func))
        if len(fsigs) > 1:
            raise QiskitError(
                "Fit functions specified in the series definition have "
                "different function signature. They should receive "
                "the same parameter set for multi-objective function fit."
            )
        fit_params = list(list(fsigs)[0].parameters.keys())[1:]

        # Validate dictionaly keys
        def _check_keys(parameter_name):
            named_values = fitter_options[parameter_name]
            if not named_values.keys() == set(fit_params):
                raise KeyError(
                    f"Fitting option {parameter_name} doesn't have the "
                    f"expected parameter names {','.join(fit_params)}."
                )

        # Convert array into dictionary
        def _dictionarize(parameter_name):
            parameter_array = fitter_options[parameter_name]
            if len(parameter_array) != len(fit_params):
                raise ValueError(
                    f"Value length of fitting option {parameter_name} doesn't "
                    "match with the length of expected parameters. "
                    f"{len(parameter_array)} != {len(fit_params)}."
                )
            return dict(zip(fit_params, parameter_array))

        if "p0" in fitter_options:
            if isinstance(fitter_options["p0"], dict):
                _check_keys("p0")
            else:
                fitter_options["p0"] = _dictionarize("p0")
        else:
            raise KeyError("Initial guess p0 is not provided to the fitting options.")

        if "bounds" in fitter_options:
            if isinstance(fitter_options["bounds"], dict):
                _check_keys("bounds")
            else:
                fitter_options["bounds"] = _dictionarize("bounds")
        else:
            fitter_options["bounds"] = dict(zip(fit_params, [(-np.inf, np.inf)] * len(fit_params)))

        return fitter_options

    def _subset_data(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_sigmas: np.ndarray,
        series: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """A helper method to extract reduced set of data.

        Args:
            name: Series name to search for.
            x_values: Full data set of x values.
            y_values: Full data set of y values.
            y_sigmas: Full data set of y sigmas.
            series: An integer array representing a mapping of data location to series index.

        Returns:
            Tuple of x values, y values, y sigmas for the specific series.

        Raises:
            QiskitError:
                - When name is not defined in the __series__ definition.
        """
        for idx, series_def in enumerate(self.__series__):
            if series_def.name == name:
                data_index = series == idx
                return x_values[data_index], y_values[data_index], y_sigmas[data_index]
        raise QiskitError(f"Specified series {name} is not defined in this analysis.")

    def _run_analysis(
        self, experiment_data: ExperimentData, **options
    ) -> Tuple[List[AnalysisResult], List["pyplot.Figure"]]:
        """Run analysis on circuit data.

        Args:
            experiment_data: the experiment data to analyze.
            options: kwarg options for analysis function.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` is a list of any
                   figures for the experiment.
        """
        analysis_result = AnalysisResult()

        # pop arguments that are not given to fitter
        plot = options.pop("plot")
        add_label = options.pop("add_label")
        axis = options.pop("ax")
        data_processor = options.pop("data_processor")
        base_fitter = options.pop("base_fitter")

        #
        # 1. Setup data processor
        #
        if isinstance(data_processor, DataProcessor) and not data_processor.is_trained:
            # Qiskit DataProcessor instance. May need calibration.
            try:
                data_processor.train(data=experiment_data.data())
            except DataProcessorError as ex:
                analysis_result["error_message"] = str(ex)
                analysis_result["success"] = False
                return [analysis_result], list()

        # get data processor options from analysis options
        processor_options = {
            key: options[key]
            for key in inspect.signature(data_processor).parameters.keys()
            if key in options
        }
        configured_data_processor = functools.partial(data_processor, **processor_options)

        #
        # 2. Extract curve entries from experiment data
        #
        # pylint: disable=broad-except
        try:
            xdata, ydata, sigma, series = self._extract_curves(
                experiment_data=experiment_data, data_processor=configured_data_processor
            )
        except Exception as ex:
            analysis_result["error_message"] = str(ex)
            analysis_result["success"] = False
            return [analysis_result], list()

        #
        # 3. Run fitting
        #
        # pylint: disable=broad-except
        try:
            # format fit data
            _xdata, _ydata, _sigma, _series = self._pre_processing(
                x_values=xdata, y_values=ydata, y_sigmas=sigma, series=series, **options
            )

            # Generate fit options
            fit_candidates = self._setup_fitting(_xdata, _ydata, _sigma, _series, **options)
            if isinstance(fit_candidates, dict):
                # only single initial guess
                fit_options = self._format_fit_options(**fit_candidates)
                fit_result = base_fitter(
                    funcs=[series_def.fit_func for series_def in self.__series__],
                    series=_series,
                    xdata=_xdata,
                    ydata=_ydata,
                    sigma=_sigma,
                    **fit_options,
                )
                analysis_result.update(**fit_result)
            else:
                # multiple initial guesses
                fit_options_candidates = [
                    self._format_fit_options(**fit_options) for fit_options in fit_candidates
                ]
                fit_results = [
                    base_fitter(
                        funcs=[series_def.fit_func for series_def in self.__series__],
                        series=_series,
                        xdata=_xdata,
                        ydata=_ydata,
                        sigma=_sigma,
                        **fit_options,
                    )
                    for fit_options in fit_options_candidates
                ]
                # Sort by chi squared value
                fit_results = sorted(fit_results, key=lambda r: r["reduced_chisq"])
                analysis_result.update(**fit_results[0])
            analysis_result["success"] = True
        except Exception as ex:
            analysis_result["error_message"] = str(ex)
            analysis_result["success"] = False

        #
        # 4. Post-process analysis data
        #
        if analysis_result["success"]:
            analysis_result = self._post_processing(analysis_result=analysis_result, **options)

        #
        # 5. Create figures
        #
        if plot:
            figures = self._create_figures(
                x_values=xdata,
                y_values=ydata,
                y_sigmas=sigma,
                series=series,
                analysis_results=analysis_result,
                axis=axis,
                add_label=add_label,
            )
        else:
            figures = list()

        #
        # 6. Save raw data
        #
        raw_data_dict = dict()
        for series_def in self.__series__:
            sub_xdata, sub_ydata, sub_sigma = self._subset_data(
                name=series_def.name, x_values=xdata, y_values=ydata, y_sigmas=sigma, series=series
            )
            raw_data_dict[series_def.name] = {
                "xdata": sub_xdata,
                "ydata": sub_ydata,
                "sigma": sub_sigma,
            }
        analysis_result["raw_data"] = raw_data_dict

        return [analysis_result], figures
