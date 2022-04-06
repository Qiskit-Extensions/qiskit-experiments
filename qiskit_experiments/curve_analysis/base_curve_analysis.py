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

"""
Base class of curve analysis.
"""

import warnings

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional

from uncertainties import unumpy as unp

from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options, ExperimentData
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.exceptions import AnalysisError

from .curve_data import CurveData, SeriesDef, FitData, ParameterRepr, FitOptions
from .curve_fit import multi_curve_fit
from .visualization import MplCurveDrawer, BaseCurveDrawer

PARAMS_ENTRY_PREFIX = "@Parameters_"
DATA_ENTRY_PREFIX = "@Data_"


class BaseCurveAnalysis(BaseAnalysis, ABC):
    """Abstract superclass of curve analysis base classes.

    Note that this class doesn't define :meth:`_run_analysis` method,
    and no actual fitting protocol is implemented by itself.
    However, this class defines several common methods that can be reused.
    A curve analysis subclass can construct proper fitting protocol
    by combining following methods, i.e. sub-routines.

    _generate_fit_guesses

        An abstract method to create initial guees.
        This should be implemented by subclass.

    _format_data

        A method to format curve data. By default this method takes y value average
        over the same x values and then sort the entire data by x values.

    _evaluate_quality

        A method to evaluate quality of fitting from fit outcome.
        This returns "good" when reduced chi-squared is less than 3.0.
        This criterion can be updated by subclass.

    _run_data_processing

        A method to perform data processing, i.e. create data arrays from
        a list of experiment data payload.

    _run_curve_fit

        A method to perform fitting with predefined fit models and formatted data.
        This method internally calls :meth:`_generate_fit_guesses`.

    _create_analysis_results

        A method to create analysis results for important fit parameters
        that might be defined by analysis options ``result_parameters``.
        In addition, another entry for all fit parameters is created when
        the analysis option ``return_fit_parameters`` is ``True``.

    _preparation

        A method that should be called before other methods are called.
        This method initializes analysis options against input experiment data.

    """

    @property
    @abstractmethod
    def parameters(self) -> List[str]:
        """Return parameters of this curve analysis."""

    @property
    def drawer(self) -> BaseCurveDrawer:
        """A short-cut for curve drawer instance."""
        return self._options.curve_plotter

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            curve_plotter (BaseCurveDrawer): A curve drawer instance to visualize
                the analysis result.
            plot_raw_data (bool): Set ``True`` to draw un-formatted data points on canvas.
                This is ``True`` by default.
            plot (bool): Set ``True`` to create figure for fit result.
                This is ``False`` by default.
            return_fit_parameters (bool): Set ``True`` to return all fit model parameters
                with details of the fit outcome. Default to ``True``.
            return_data_points (bool): Set ``True`` to return formatted data points.
                Default to ``False``.
            curve_fitter (Callable): A callback function to perform fitting with formatted data.
                See :func:`~qiskit_experiments.analysis.multi_curve_fit` for example.
            data_processor (Callable): A callback function to format experiment data.
                This can be a :class:`~qiskit_experiments.data_processing.DataProcessor`
                instance that defines the `self.__call__` method.
            normalization (bool) : Set ``True`` to normalize y values within range [-1, 1].
            p0 (Dict[str, float]): Array-like or dictionary
                of initial parameters.
            bounds (Dict[str, Tuple[float, float]]): Array-like or dictionary
                of (min, max) tuple of fit parameter boundaries.
            x_key (str): Circuit metadata key representing a scanned value.
            result_parameters (List[Union[str, ParameterRepr]): Parameters reported in the
                database as a dedicated entry. This is a list of parameter representation
                which is either string or ParameterRepr object. If you provide more
                information other than name, you can specify
                ``[ParameterRepr("alpha", "\u03B1", "a.u.")]`` for example.
                The parameter name should be defined in the series definition.
                Representation should be printable in standard output, i.e. no latex syntax.
            extra (Dict[str, Any]): A dictionary that is appended to all database entries
                as extra information.
            curve_fitter_options (Dict[str, Any]) Options that are passed to the
                specified curve fitting function.
            fixed_parameters (Dict[str, Any]): Fitting model parameters that are fixed
                during the curve fitting. This should be provided with default value
                keyed on one of the parameter names in the series definition.
            chisq_threshold (float):
        """
        options = super()._default_options()

        options.curve_plotter = MplCurveDrawer()
        options.plot_raw_data = False
        options.plot = True
        options.return_fit_parameters = True
        options.return_data_points = False
        options.curve_fitter = multi_curve_fit
        options.data_processor = None
        options.normalization = False
        options.x_key = "xval"
        options.result_parameters = []
        options.extra = {}
        options.curve_fitter_options = {}
        options.p0 = {}
        options.bounds = {}
        options.fixed_parameters = {}

        return options

    def set_options(self, **fields):
        """Set the analysis options for :meth:`run` method.

        Args:
            fields: The fields to update the options

        Raises:
            KeyError: When removed option ``curve_fitter`` is set.
            TypeError: When invalid drawer instance is provided.
        """
        # TODO remove this in Qiskit Experiments v0.4
        if "curve_plotter" in fields and isinstance(fields["curve_plotter"], str):
            plotter_str = fields["curve_plotter"]
            warnings.warn(
                f"The curve plotter '{plotter_str}' has been deprecated. "
                "The option is replaced with 'MplCurveDrawer' instance. "
                "If this is a loaded analysis, please save this instance again to update option value. "
                "This warning will be removed with backport in Qiskit Experiments 0.4.",
                DeprecationWarning,
                stacklevel=2,
            )
            fields["curve_plotter"] = MplCurveDrawer()

        if "curve_plotter" in fields and not isinstance(fields["curve_plotter"], BaseCurveDrawer):
            plotter_obj = fields["curve_plotter"]
            raise TypeError(
                f"'{plotter_obj.__class__.__name__}' object is not valid curve drawer instance."
            )

        # pylint: disable=no-member
        draw_options = set(self.drawer.options.__dict__.keys()) | {"style"}
        deprecated = draw_options & fields.keys()
        if any(deprecated):
            warnings.warn(
                f"Option(s) {deprecated} have been moved to draw_options and will be removed soon. "
                "Use self.drawer.set_options instead. "
                "If this is a loaded analysis, please save this instance again to update option value. "
                "This warning will be removed with backport in Qiskit Experiments 0.4.",
                DeprecationWarning,
                stacklevel=2,
            )
            draw_options = dict()
            for depopt in deprecated:
                if depopt == "style":
                    for k, v in fields.pop("style").items():
                        draw_options[k] = v
                else:
                    draw_options[depopt] = fields.pop(depopt)
            self.drawer.set_options(**draw_options)

        super().set_options(**fields)

    @abstractmethod
    def _generate_fit_guesses(
        self,
        user_opt: FitOptions,
        curve_data: CurveData,
    ) -> Union[FitOptions, List[FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """

    def _format_data(
        self,
        curve_data: CurveData,
    ) -> CurveData:
        """Post-processing for fit data collection.

        Args:
            curve_data: Raw data collection created from experiment results.

        Returns:
            Formatted data.
        """
        # take average over the same x value by keeping sigma
        data_allocation, xdata, ydata, sigma, shots = multi_mean_xy_data(
            series=curve_data.data_allocation,
            xdata=curve_data.x,
            ydata=curve_data.y,
            sigma=curve_data.y_err,
            shots=curve_data.shots,
            method="shots_weighted",
        )

        # sort by x value in ascending order
        data_allocation, xdata, ydata, sigma, shots = data_sort(
            series=data_allocation,
            xdata=xdata,
            ydata=ydata,
            sigma=sigma,
            shots=shots,
        )

        return CurveData(
            x=xdata,
            y=ydata,
            y_err=sigma,
            shots=shots,
            data_allocation=data_allocation,
            labels=curve_data.labels,
        )

    def _evaluate_quality(
        self,
        fit_data: FitData,
    ) -> Union[str, None]:
        """Evaluate quality of the fit result.

        Args:
            fit_data: Fit outcome.

        Returns:
            String that represents fit result quality. Usually "good" or "bad".
        """
        if fit_data.reduced_chisq < 3.0:
            return "good"
        return "bad"

    def _run_data_processing(
        self,
        raw_data: List[Dict],
        series: List[SeriesDef],
    ) -> CurveData:
        """Perform data processing from the experiment result payload.

        Args:
            raw_data: Payload in the experiment data.
            series: List of series definition defining filtering condition.

        Returns:
            Un-formatted data collection.
        """
        x_key = self.options.x_key

        try:
            xdata = np.asarray([datum["metadata"][x_key] for datum in data], dtype=float)
        except KeyError as ex:
            raise DataProcessorError(
                f"X value key {x_key} is not defined in circuit metadata."
            ) from ex

        ydata = self.options.data_processor(data)
        shots = np.asarray([datum.get("shots", np.nan) for datum in data])

        def _matched(metadata, **filters):
            try:
                return all(metadata[key] == val for key, val in filters.items())
            except KeyError:
                return False

        data_allocation = np.full(xdata.size, -1, dtype=int)
        for sind, series_def in enumerate(series):
            matched_inds = np.asarray(
                [_matched(d["metadata"], **series_def.filter_kwargs) for d in data], dtype=bool
            )
            data_allocation[matched_inds] = sind

        return CurveData(
            x=xdata,
            y=unp.nominal_values(ydata),
            y_err=unp.std_devs(ydata),
            shots=shots,
            data_allocation=data_allocation,
            labels=[s.name for s in series],
        )

    def _run_curve_fit(
        self,
        curve_data: CurveData,
        series: List[SeriesDef],
    ) -> Union[None, FitData]:
        """Perform curve fitting on given data collection and fit models.

        Args:
            curve_data: A formatted data collection to fit.
            series: A list of fit models.

        Returns:
            The best fitting outcome with minimum reduced chi-squared value.
        """
        # Create a list of initial guess
        default_fit_opt = FitOptions(
            parameters=self.parameters,
            default_p0=self.options.p0,
            default_bounds=self.options.bounds,
            **self.options.curve_fitter_options,
        )
        try:
            fit_options = self._generate_fit_guesses(default_fit_opt, curve_data)
        except TypeError:
            warnings.warn(
                "Calling '_generate_fit_guesses' method without curve data has been "
                "deprecated and will be prohibited after 0.4. "
                "Update the method signature of your custom analysis class.",
                DeprecationWarning,
            )
            fit_options = self._generate_fit_guesses(default_fit_opt)
        if isinstance(fit_options, FitOptions):
            fit_options = [fit_options]

        # Run fit for each configuration
        fit_results = []
        for fit_opt in set(fit_options):
            try:
                fit_result = self.options.curve_fitter(
                    funcs=[sdef.fit_func for sdef in series],
                    series=curve_data.data_index,
                    xdata=curve_data.x,
                    ydata=curve_data.y,
                    sigma=curve_data.y_err,
                    **fit_opt.options,
                )
                fit_results.append(fit_result)
            except AnalysisError:
                # Some guesses might be too far from the true parameters and may thus fail.
                # We ignore initial guesses that fail and continue with the next fit candidate.
                pass

        # Find best value with chi-squared value
        if len(fit_results) == 0:
            warnings.warn(
                "All initial guesses and parameter boundaries failed to fit the data. "
                "Please provide better initial guesses or fit parameter boundaries.",
                UserWarning,
            )
            # at least return raw data points rather than terminating
            return None

        return sorted(fit_results, key=lambda r: r.reduced_chisq)[0]

    def _create_analysis_results(
        self,
        fit_data: FitData,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for important fit parameters.

        Args:
            fit_data: Fit outcome.

        Returns:
            List of analysis result data.
        """
        quality = self._evaluate_quality(fit_data=fit_data)
        outcomes = []

        # Create entry for all fit parameters
        if self.options.return_fit_parameters:
            fit_parameters = AnalysisResultData(
                name=PARAMS_ENTRY_PREFIX + self.__class__.__name__,
                value=[p.nominal_value for p in fit_data.popt],
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra={
                    "popt_keys": fit_result.popt_keys,
                    "dof": fit_result.dof,
                    "covariance_mat": fit_result.pcov,
                    "fit_models": fit_models,
                    **metadata,
                },
            )
            outcomes.append(fit_parameters)

        # Create entries for important parameters
        for param_repr in self.options.result_parameters:
            if isinstance(param_repr, ParameterRepr):
                p_name = param_repr.name
                p_repr = param_repr.repr or param_repr.name
                unit = param_repr.unit
            else:
                p_name = param_repr
                p_repr = param_repr
                unit = None

            fit_val = fit_data.fitval(p_name)
            if unit:
                par_metadata = metadata.copy()
                par_metadata["unit"] = unit
            else:
                par_metadata = metadata

            outcome = AnalysisResultData(
                name=p_repr,
                value=fit_val,
                chisq=fit_result.reduced_chisq,
                quality=quality,
                extra=par_metadata,
            )
            outcomes.append(outcome)

        return outcomes

    def _preparation(
        self,
        experiment_data: ExperimentData,
    ):
        """Prepare for curve analysis. This method is called ahead of other processing.

        Args:
            experiment_data: Experiment data to analyze.
        """
        # Initialize canvas
        if self.options.plot:
            self.drawer.initialize_canvas()

        # Initialize data processor
        # TODO move this to base analysis in follow-up
        data_processor = self.options.data_processor or get_processor(experiment_data, self.options)

        if isinstance(data_processor, DataProcessor):
            if not data_processor.is_trained:
                data_processor.train(data=experiment_data.data())
            self.set_options(data_processor=data_processor)
        else:
            raise AnalysisError(
                f"'{repr(data_processor)}' is not valid data processor. "
                "Please provide DataProcessor subclass in the analysis option."
            )
