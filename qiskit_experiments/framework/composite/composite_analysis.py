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

from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from qiskit.result import marginal_distribution
from qiskit.result.postprocess import format_counts_memory
from qiskit_experiments.framework import BaseAnalysis, ExperimentData
from qiskit_experiments.framework.analysis_result_data import AnalysisResultData
from qiskit_experiments.framework.base_analysis import _requires_copy
from qiskit_experiments.exceptions import AnalysisError


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
        flatten_results: bool = True,
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

    def run(
        self,
        experiment_data: ExperimentData,
        replace_results: bool = False,
        **options,
    ) -> ExperimentData:
        # Make a new copy of experiment data if not updating results
        if not replace_results and _requires_copy(experiment_data):
            experiment_data = experiment_data.copy()

        if not self._flatten_results:
            # Initialize child components if they are not initialized
            # This only needs to be done if results are not being flattened
            self._add_child_data(experiment_data)

        # Run analysis with replace_results = True since we have already
        # created the copy if it was required
        return super().run(experiment_data, replace_results=True, **options)

    def _run_analysis(self, experiment_data: ExperimentData):
        # Return list of experiment data containers for each component experiment
        # containing the marginalized data from the composite experiment
        component_expdata = self._component_experiment_data(experiment_data)

        # Run the component analysis on each component data
        for i, sub_expdata in enumerate(component_expdata):
            # Since copy for replace result is handled at the parent level
            # we always run with replace result on component analysis
            self._analyses[i].run(sub_expdata, replace_results=True)
            # Block for results to avoid issues nested CompositeAnalysis runs.
            # Ideally this constraint will be removed in the future.
            # Previously run() was called on all components and then
            # block_for_results() was called on all of them, but since only one
            # thread can execute Python code at a time there is not much
            # difference in performance. Blocking separately limits the number
            # of threads that are started simultaneously.
            sub_expdata.block_for_results()

        # Optionally flatten results from all component experiments
        # for adding to the main experiment data container
        if self._flatten_results:
            analysis_results, figures = self._combine_results(component_expdata)
            for res in analysis_results:
                # Override experiment  ID because entries are flattened
                res.experiment_id = experiment_data.experiment_id
            return analysis_results, figures
        return [], []

    def _component_experiment_data(self, experiment_data: ExperimentData) -> List[ExperimentData]:
        """Return a list of marginalized experiment data for component experiments.

        Args:
            experiment_data: a composite experiment data container.

        Returns:
            The list of analysis-ready marginalized experiment data for each
            component experiment.

        Raises:
            AnalysisError: If the component experiment data cannot be extracted.
        """
        if not self._flatten_results:
            # Retrieve child data for component experiments for updating
            component_index = experiment_data.metadata.get("component_child_index", [])
            if not component_index:
                raise AnalysisError("Unable to extract component child experiment data")
            component_expdata = [experiment_data.child_data(i) for i in component_index]
        else:
            # Initialize temporary ExperimentData containers for
            # each component experiment to analysis on. These will
            # not be saved but results and figures will be collected
            # from them
            component_expdata = self._initialize_component_experiment_data(experiment_data)

        # Compute marginalize data for each component experiment
        marginalized_data = self._marginalized_component_data(experiment_data.data())

        # Add the marginalized component data and component job metadata
        # to each component child experiment. Note that this will clear
        # any currently stored data in the experiment. Since copying of
        # child data is handled by the `replace_results` kwarg of the
        # parent container it is safe to always clear and replace the
        # results of child containers in this step
        for sub_expdata, sub_data in zip(component_expdata, marginalized_data):
            # Clear any previously stored data and add marginalized data
            sub_expdata._result_data.clear()
            sub_expdata.add_data(sub_data)

        return component_expdata

    def _marginalized_component_data(self, composite_data: List[Dict]) -> List[List[Dict]]:
        """Return marginalized data for component experiments.

        Args:
            composite_data: a list of composite experiment circuit data.

        Returns:
            A List of lists of marginalized circuit data for each component
            experiment in the composite experiment.
        """
        # Marginalize data
        marginalized_data = {}
        for datum in composite_data:
            metadata = datum.get("metadata", {})

            # Add marginalized data to sub experiments
            if "composite_clbits" in metadata:
                composite_clbits = metadata["composite_clbits"]
            else:
                composite_clbits = None

            # Pre-process the memory if any to avoid redundant calls to format_counts_memory
            f_memory = self._format_memory(datum, composite_clbits)

            for i, index in enumerate(metadata["composite_index"]):
                if index not in marginalized_data:
                    # Initialize data list for marginalized
                    marginalized_data[index] = []
                sub_data = {
                    k: v for k, v in datum.items() if k not in ("metadata", "counts", "memory")
                }
                sub_data["metadata"] = metadata["composite_metadata"][i]
                if "counts" in datum:
                    if composite_clbits is not None:
                        sub_data["counts"] = marginal_distribution(
                            counts=datum["counts"],
                            indices=composite_clbits[i],
                        )
                    else:
                        sub_data["counts"] = datum["counts"]
                if "memory" in datum:
                    if composite_clbits is not None:
                        # level 2
                        if f_memory is not None:
                            idx = slice(
                                -1 - composite_clbits[i][-1], -composite_clbits[i][0] or None
                            )
                            sub_data["memory"] = [shot[idx] for shot in f_memory]
                        # level 1
                        else:
                            mem = np.array(datum["memory"])

                            # Averaged level 1 data
                            if len(mem.shape) == 2:
                                sub_data["memory"] = mem[composite_clbits[i]].tolist()
                            # Single-shot level 1 data
                            if len(mem.shape) == 3:
                                sub_data["memory"] = mem[:, composite_clbits[i]].tolist()
                    else:
                        sub_data["memory"] = datum["memory"]
                marginalized_data[index].append(sub_data)

        # Sort by index
        return [marginalized_data[i] for i in sorted(marginalized_data.keys())]

    @staticmethod
    def _format_memory(datum: Dict, composite_clbits: List):
        """A helper method to convert level 2 memory (if it exists) to bit-string format."""
        f_memory = None
        if (
            "memory" in datum
            and composite_clbits is not None
            and isinstance(datum["memory"][0], str)
        ):
            num_cbits = 1 + max(cbit for cbit_list in composite_clbits for cbit in cbit_list)
            header = {"memory_slots": num_cbits}
            f_memory = list(format_counts_memory(shot, header) for shot in datum["memory"])

        return f_memory

    def _add_child_data(self, experiment_data: ExperimentData):
        """Save empty component experiment data as child data.

        This will initialize empty ExperimentData objects for each component
        experiment and add them as child data to the main composite experiment
        ExperimentData container container for saving.

        Args:
            experiment_data: a composite experiment experiment data container.
        """
        component_index = experiment_data.metadata.get("component_child_index", [])
        if component_index:
            # Child components are already initialized
            return

        # Initialize the component experiment data containers and add them
        # as child data to the current experiment data
        child_components = self._initialize_component_experiment_data(experiment_data)
        start_index = len(experiment_data.child_data())
        for i, subdata in enumerate(child_components):
            experiment_data.add_child_data(subdata)
            component_index.append(start_index + i)

        # Store the indices of the added child data in metadata
        experiment_data.metadata["component_child_index"] = component_index

    def _initialize_component_experiment_data(
        self, experiment_data: ExperimentData
    ) -> List[ExperimentData]:
        """Initialize empty experiment data containers for component experiments.

        Args:
            experiment_data: a composite experiment experiment data container.

        Returns:
            The list of experiment data containers for each component experiment
            containing the component metadata, and tags, share level, and
            auto save settings of the composite experiment.
        """
        # Extract component experiment types and metadata so they can be
        # added to the component experiment data containers
        metadata = experiment_data.metadata
        num_components = len(self._analyses)
        experiment_types = metadata.get("component_types", [None] * num_components)
        component_metadata = metadata.get("component_metadata", [{}] * num_components)

        # Create component experiments and set the backend and
        # metadata for the components
        component_expdata = []
        for i, _ in enumerate(self._analyses):
            subdata = ExperimentData(backend=experiment_data.backend)
            subdata.experiment_type = experiment_types[i]
            subdata.metadata.update(component_metadata[i])

            if self._flatten_results:
                # Explicitly set auto_save to false so the temporary
                # data can't accidentally be saved
                subdata.auto_save = False
            else:
                # Copy tags, share_level and auto_save from the parent
                # experiment data if results are not being flattened.
                subdata.tags = experiment_data.tags
                subdata.share_level = experiment_data.share_level
                subdata.auto_save = experiment_data.auto_save

            component_expdata.append(subdata)

        return component_expdata

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
            for artifact in sub_expdata.artifacts():
                analysis_results.append(artifact)

        return analysis_results, figures
