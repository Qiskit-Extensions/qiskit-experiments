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
Composite Experiment Analysis class.
"""

from typing import List, Union, Optional, Tuple
import warnings
from qiskit_experiments.framework import BaseAnalysis, ExperimentData
from qiskit_experiments.framework.analysis_result_data import AnalysisResultData


class CompositeAnalysis(BaseAnalysis):
    """Run analysis for composite experiments.

    Composite experiments consist of several component experiments
    run together in a single execution, the results of which are returned
    as a single list of circuit result data in the :class:`ExperimentData`
    container.

    Analysis of this composite circuit data involves constructing
    a list of experiment data containers for each component experiment containing
    the marginalized circuit result data for that experiment. These are saved as
    :meth:`~.ExperimentData.child_data` in the main :class:`.ExperimentData` container.
    Each component experiment data is then analyzed using the analysis class from
    the corresponding component experiment.

    .. note::

        If the composite :class:`ExperimentData` does not already contain child
        experiment data containers for the component experiments they will be
        initialized and added to the experiment data when
        :meth:`~.CompositeAnalysis.run` is called on the composite data.

        When calling :meth:`~.CompositeAnalysis.run` on experiment data already
        containing initialized component experiment data, any previously stored circuit
        data will be cleared and replaced with the marginalized data from the composite
        experiment data.
    """

    def __init__(
        self,
        analyses: List[BaseAnalysis],
        flatten_results: bool = None,
        generate_figures: Optional[str] = "always",
    ):
        """Initialize a composite analysis class.

        Args:
            analyses: a list of component experiment analysis objects.
            flatten_results: If True flatten all component experiment results
                             into a single ExperimentData container, including
                             nested composite experiments. If False save each
                             component experiment results as a separate child
                             ExperimentData container.
            generate_figures: Optional flag to set the figure generation behavior.
                If ``always``, figures are always generated. If ``never``, figures are never generated.
                If ``selective``, figures are generated if the analysis ``quality`` is ``bad``.
        """
        if flatten_results is None:
            # Backward compatibility for 0.6
            # This if-clause will be removed in 0.7 and flatten_result=True is set in arguments.
            warnings.warn(
                "Default value of flatten_results will be turned to True in Qiskit Experiments 0.7. "
                "If you want child experiment data for each subset experiment, "
                "set 'flatten_results=False' explicitly.",
                DeprecationWarning,
            )
            flatten_results = False
        super().__init__()
        self._analyses = analyses
        self._flatten_results = False
        if flatten_results:
            self._set_flatten_results()

        self._set_generate_figures(generate_figures)

    def component_analysis(
        self, index: Optional[int] = None
    ) -> Union[BaseAnalysis, List[BaseAnalysis]]:
        """Return the component experiment Analysis instance.

        Args:
            index: Optional, the component index to return analysis for.
                   If None return a list of all component analysis instances.

        Returns:
            The analysis instance for the specified index, or a list of all
            analysis instances if index is None.
        """
        if index is None:
            return self._analyses
        return self._analyses[index]

    def set_options(self, **fields):
        """Set the analysis options for the experiment. If the `broadcast` argument set to `True`, the
        analysis options will cascade to the child experiments."""
        super().set_options(**fields)
        if fields.get("broadcast", None):
            for sub_analysis in self._analyses:
                sub_analysis.set_options(**fields)

    def copy(self):
        ret = super().copy()
        # Recursively copy analysis
        ret._analyses = [analysis.copy() for analysis in ret._analyses]
        return ret

    def _run_analysis(self, experiment_data: ExperimentData):
        child_data = experiment_data.child_data()

        if len(self._analyses) != len(child_data):
            # Child data is automatically created when composite result data is added.
            # Validate that child data size matches with number of analysis entries.
            experiment_data.create_child_data()

        for sub_analysis, sub_data in zip(self._analyses, child_data):
            # Since copy for replace result is handled at the parent level
            # we always run with replace result on component analysis
            sub_analysis.run(sub_data, replace_results=True)

        # Analysis is running in parallel so we add loop to wait
        # for all component analysis to finish before returning
        # the parent experiment analysis results
        for sub_data in child_data:
            sub_data.block_for_results()

        # Optionally flatten results from all component experiments
        # for adding to the main experiment data container
        if self._flatten_results:
            analysis_results, figures = self._combine_results(child_data)

            for res in analysis_results:
                # Override experiment  ID because entries are flattened
                res.experiment_id = experiment_data.experiment_id
            return analysis_results, figures
        return [], []

    def _set_flatten_results(self):
        """Recursively set flatten_results to True for all composite components."""
        self._flatten_results = True
        for analysis in self._analyses:
            if isinstance(analysis, CompositeAnalysis):
                analysis._set_flatten_results()

    def _set_generate_figures(self, generate_figures):
        """Recursively propagate ``generate_figures`` to all child experiments."""
        self._generate_figures = generate_figures
        for analysis in self._analyses:
            if isinstance(analysis, CompositeAnalysis):
                analysis._set_generate_figures(generate_figures)
            else:
                analysis._generate_figures = generate_figures

    def _combine_results(
        self,
        component_experiment_data: List[ExperimentData],
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Combine analysis results from component experiment data.

        Args:
            component_experiment_data: list of experiment data containers containing the
                                       analysis results for each component experiment.

        Returns:
            A pair of the combined list of all analysis results from each of the
            component experiments, and a list of all figures from each component
            experiment.
        """
        analysis_results = []
        figures = []
        for sub_expdata in component_experiment_data:
            figures += sub_expdata._figures.values()

            # Convert Dataframe Series back into AnalysisResultData
            # This is due to limitation that _run_analysis must return List[AnalysisResultData],
            # and some composite analysis such as TphiAnalysis overrides this method to
            # return extra quantity computed from sub analysis results.
            # This produces unnecessary data conversion.
            # The _run_analysis mechanism seems just complicating the entire logic.
            # Since it's impossible to deprecate the usage of this protected method,
            # we should implement new CompositeAnalysis class with much more efficient
            # internal logic. Note that the child data structure is no longer necessary
            # because dataframe offers more efficient data filtering mechanisms.
            analysis_table = sub_expdata.analysis_results(columns="all", dataframe=True)
            for _, series in analysis_table.iterrows():
                data = AnalysisResultData.from_table_element(**series.to_dict())
                analysis_results.append(data)

        return analysis_results, figures
