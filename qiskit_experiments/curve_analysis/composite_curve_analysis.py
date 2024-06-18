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
Analysis class for multi-group curve fitting.
"""
# pylint: disable=invalid-name
import warnings
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import lmfit
import numpy as np
import pandas as pd

from qiskit.utils.deprecation import deprecate_func

from qiskit_experiments.framework import (
    AnalysisResultData,
    BaseAnalysis,
    ExperimentData,
    Options,
)
from qiskit_experiments.visualization import (
    BaseDrawer,
    BasePlotter,
    CurvePlotter,
    LegacyCurveCompatDrawer,
    MplDrawer,
)

from qiskit_experiments.framework.containers import FigureType, ArtifactData
from .base_curve_analysis import BaseCurveAnalysis
from .curve_data import CurveFitResult
from .scatter_table import ScatterTable


class CompositeCurveAnalysis(BaseAnalysis):
    r"""Composite Curve Analysis.

    The :class:`.CompositeCurveAnalysis` takes multiple curve analysis instances
    and performs each analysis on the same experimental results.
    These analyses are performed independently, thus fit parameters have no correlation.
    Note that this is different from :class:`.CompositeAnalysis` which
    analyses the outcome of a composite experiment, in which multiple different
    experiments are performed.
    The :class:`.CompositeCurveAnalysis` is attached to a single experiment instance,
    which may execute similar circuits with slightly different settings.
    Experiments with different settings might be distinguished by the circuit
    metadata. The outcomes of the same set of experiments are assigned to a
    specific analysis instance in the composite curve analysis.
    This mapping is usually done with the analysis option ``filter_data`` dictionary.
    Otherwise, all analyses are performed on the same set of outcomes.

    Examples:

        In this example, we write up a composite analysis consisting of two oscillation
        analysis instances, assuming two Rabi experiments in 1-2 subspace
        starting with different initial states :math:`\in \{|0\rangle, |1\rangle\}`.
        This is a typical procedure to measure the thermal population of the qubit.

        .. code-block:: python

            from qiskit_experiments import curve_analysis as curve

            analyses = []
            for qi in (0, 1):
                analysis = curve.OscillationAnalysis(name=f"init{qi}")
                analysis.set_options(
                    return_fit_parameters=["freq"],
                    filter_data={"init_state": qi},
                )
            analysis = CompositeCurveAnalysis(analyses=analyses)

        This ``analysis`` will return two analysis result data for the fit parameter "freq"
        for experiments with the initial state :math:`|0\rangle` and :math:`|1\rangle`.
        The experimental circuits starting with different initial states must be
        distinguished by the circuit metadata ``{"init_state": 0}`` or ``{"init_state": 1}``,
        along with the "xval" in the same dictionary.

    :class:`.CompositeCurveAnalysis` subclass may override following methods.

    .. rubric:: _evaluate_quality

    This method evaluates the quality of the composite fit based on
    the all analysis outcomes.
    This returns "good" when all fit outcomes are evaluated as "good",
    otherwise it returns "bad".

    .. rubric:: _create_analysis_results

    This method is passed all the group fit outcomes and can return a list of
    new values to be stored in the analysis results.

    .. rubric:: _create_figures

    This method creates figures by consuming the scatter table data.
    Figures are created when the analysis option ``plot`` is ``True``.

    """

    def __init__(
        self,
        analyses: List[BaseCurveAnalysis],
        name: Optional[str] = None,
    ):
        super().__init__()

        self._analyses = analyses
        self._name = name or self.__class__.__name__

    @property
    def parameters(self) -> List[str]:
        """Return parameters of this curve analysis."""
        unite_params = []
        for analysis in self._analyses:
            # Respect ordering of parameters
            for name in analysis.parameters:
                if name not in unite_params:
                    unite_params.append(name)
        return unite_params

    @property
    def name(self) -> str:
        """Return name of this analysis."""
        return self._name

    @property
    def models(self) -> Dict[str, List[lmfit.Model]]:
        """Return fit models."""
        models = {}
        for analysis in self._analyses:
            models[analysis.name] = analysis.models
        return models

    @property
    def plotter(self) -> BasePlotter:
        """A short-cut to the plotter instance."""
        return self._options.plotter

    @property
    @deprecate_func(
        since="0.5",
        additional_msg="Use `plotter` from the new visualization module instead.",
        removal_timeline="after 0.6",
        package_name="qiskit-experiments",
    )
    def drawer(self) -> BaseDrawer:
        """A short-cut for curve drawer instance, if set. ``None`` otherwise."""
        if hasattr(self._options, "curve_drawer"):
            return self._options.curve_drawer
        else:
            return None

    def analyses(
        self, index: Optional[Union[str, int]] = None
    ) -> Union[BaseCurveAnalysis, List[BaseCurveAnalysis]]:
        """Return curve analysis instance.

        Args:
            index: Name of group or numerical index.

        Returns:
            Curve analysis instance.
        """
        if index is None:
            return self._analyses
        if isinstance(index, str):
            group_names = [analysis.name for analysis in self._analyses]
            num_index = group_names.index(index)
            return self._analyses[num_index]
        return self._analyses[index]

    def _evaluate_quality(
        self,
        fit_data: Dict[str, CurveFitResult],
    ) -> Union[str, None]:
        """Evaluate quality of the fit result.

        Args:
            fit_data: Fit outcome keyed on the analysis name.

        Returns:
            String that represents fit result quality. Usually "good" or "bad".
        """
        for analysis in self._analyses:
            if analysis._evaluate_quality(fit_data[analysis.name]) != "good":
                return "bad"
        return "good"

    # pylint: disable=unused-argument
    def _create_analysis_results(
        self,
        fit_data: Dict[str, CurveFitResult],
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results based on all analysis outcomes.

        Args:
            fit_data: Fit outcome keyed on the analysis name.
            quality: Quality of fit outcome.

        Returns:
            List of analysis result data.
        """
        return []

    def _create_figures(
        self,
        curve_data: ScatterTable,
    ) -> List["matplotlib.figure.Figure"]:
        """Create a list of figures from the curve data.

        Args:
            curve_data: Scatter data table containing all data points.

        Returns:
            A list of figures.
        """
        for analysis in self.analyses():
            group_data = curve_data.filter(analysis=analysis.name)
            model_names = analysis.model_names()
            for series_id, sub_data in group_data.iter_by_series_id():
                full_name = f"{model_names[series_id]}_{analysis.name}"
                # Plot raw data scatters
                if analysis.options.plot_raw_data:
                    raw_data = sub_data.filter(category="raw")
                    self.plotter.set_series_data(
                        series_name=full_name,
                        x=raw_data.x,
                        y=raw_data.y,
                    )
                # Plot formatted data scatters
                formatted_data = sub_data.filter(category=analysis.options.fit_category)
                self.plotter.set_series_data(
                    series_name=full_name,
                    x_formatted=formatted_data.x,
                    y_formatted=formatted_data.y,
                    y_formatted_err=formatted_data.y_err,
                )
                # Plot fit lines
                line_data = sub_data.filter(category="fitted")
                if len(line_data) == 0:
                    continue
                fit_stdev = line_data.y_err
                self.plotter.set_series_data(
                    series_name=full_name,
                    x_interp=line_data.x,
                    y_interp=line_data.y,
                    y_interp_err=fit_stdev if np.isfinite(fit_stdev).all() else None,
                )

        return [self.plotter.figure()]

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options.

        Analysis Options:
            plotter (BasePlotter): A plotter instance to visualize
                the analysis result.
            plot (bool): Set ``True`` to create figure for fit result.
                This is ``True`` by default.
            return_fit_parameters (bool): (Deprecated) Set ``True`` to return all fit model parameters
                with details of the fit outcome. Default to ``False``.
            return_data_points (bool): (Deprecated) Set ``True`` to include in the analysis result
                the formatted data points given to the fitter. Default to ``False``.
            extra (Dict[str, Any]): A dictionary that is appended to all database entries
                as extra information.
        """
        options = super()._default_options()
        options.update_options(
            plotter=CurvePlotter(MplDrawer()),
            plot=True,
            return_fit_parameters=False,
            return_data_points=False,
            extra={},
        )

        # Set automatic validator for particular option values
        options.set_validator(field="plotter", validator_value=BasePlotter)

        return options

    def set_options(self, **fields):
        # TODO remove this in Qiskit Experiments 0.6
        if "curve_drawer" in fields:
            warnings.warn(
                "The option 'curve_drawer' is replaced with 'plotter'. "
                "This option will be removed in Qiskit Experiments 0.6.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Set the plotter drawer to `curve_drawer`. If `curve_drawer` is the right type, set it
            # directly. If not, wrap it in a compatibility drawer.
            if isinstance(fields["curve_drawer"], BaseDrawer):
                plotter = self.options.plotter
                plotter.drawer = fields.pop("curve_drawer")
                fields["plotter"] = plotter
            else:
                drawer = fields["curve_drawer"]
                compat_drawer = LegacyCurveCompatDrawer(drawer)
                plotter = self.options.plotter
                plotter.drawer = compat_drawer
                fields["plotter"] = plotter

        for field in fields:
            if not hasattr(self.options, field):
                warnings.warn(
                    f"Specified option {field} doesn't exist in this analysis instance. "
                    f"Note that {self.__class__.__name__} is a composite curve analysis instance, "
                    "which consists of multiple child curve analyses. "
                    "This options may exist in each analysis instance. "
                    "Please try setting options to child analyses through '.analyses()'.",
                    UserWarning,
                )
        super().set_options(**fields)

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[Union[AnalysisResultData, ArtifactData]], List[FigureType]]:
        result_data: List[Union[AnalysisResultData, ArtifactData]] = []
        figures: List[FigureType] = []
        artifacts: list[ArtifactData] = []

        # Flag for plotting can be "always", "never", or "selective"
        # the analysis option overrides self._generate_figures if set
        if self.options.get("plot", None):
            plot = "always"
        elif self.options.get("plot", None) is False:
            plot = "never"
        else:
            plot = getattr(self, "_generate_figures", "always")

        sub_artifacts = defaultdict(list)
        for source_analysis in self._analyses:
            analysis = source_analysis.copy()
            metadata = analysis.options.extra
            metadata["group"] = analysis.name
            analysis.set_options(
                plot=False,
                extra=metadata,
                return_fit_parameters=self.options.return_fit_parameters,
                return_data_points=self.options.return_data_points,
            )
            results, _ = analysis._run_analysis(experiment_data)
            for res in results:
                if isinstance(res, ArtifactData):
                    sub_artifacts[res.name].append((analysis.name, res.data))
                else:
                    result_data.append(res)

        if "curve_data" in sub_artifacts:
            combined_curve_data = ScatterTable.from_dataframe(
                data=pd.concat([d.dataframe for _, d in sub_artifacts["curve_data"]])
            )
            artifacts.append(ArtifactData(name="curve_data", data=combined_curve_data))
        else:
            combined_curve_data = None

        if "fit_summary" in sub_artifacts:
            combined_summary = dict(sub_artifacts["fit_summary"])
            artifacts.append(ArtifactData(name="fit_summary", data=combined_summary))
            total_quality = self._evaluate_quality(combined_summary)
        else:
            combined_summary = None
            total_quality = "No Information"

        # After the quality is determined, plot can become a boolean flag for whether
        # to generate the figure
        plot_bool = plot == "always" or (plot == "selective" and total_quality == "bad")

        # Create analysis results by combining all fit data
        if combined_summary and all(fit_data.success for fit_data in combined_summary.values()):
            composite_results = self._create_analysis_results(
                fit_data=combined_summary,
                quality=total_quality,
                **self.options.extra.copy(),
            )
            result_data.extend(composite_results)
        else:
            composite_results = []

        if plot_bool and combined_curve_data:
            if combined_summary:
                red_chi_dict = {
                    k: v.reduced_chisq for k, v in combined_summary.items() if v.success
                }
            else:
                red_chi_dict = {}
            self.plotter.set_supplementary_data(
                fit_red_chi=red_chi_dict,
                primary_results=composite_results,
            )
            figures.extend(self._create_figures(curve_data=combined_curve_data))

        return result_data + artifacts, figures
