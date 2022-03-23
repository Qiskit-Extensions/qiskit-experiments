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
from qiskit.result import marginal_counts
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
    :meth:.~ExperimentData.child_data` in the main :class:`.ExperimentData` container.
    Each component experiment data is then analyzed using the analysis class from
    the corresponding component experiment.

    .. note::

        If the composite :class:`ExperimentData` does not already contain
        child experiment data containers for the component experiments
        they will be initialized and added to the experiment data when :meth:`run`
        is called on the composite data.

        When calling :meth:`run` on experiment data already containing
        initialized component experiment data, any previously stored
        circuit data will be cleared and replaced with the marginalized data
        from the composite experiment data.
    """

    def __init__(self, analyses: List[BaseAnalysis], **options):
        """Initialize a composite analysis class.

        Args:
            analyses: a list of component experiment analysis objects.
            options: set values for analysis options.
        """
        super().__init__()
        self._analyses = analyses

        # Set any init kwarg analysis options
        if options:
            self.set_options(**options)

    @classmethod
    def _default_options(cls):
        """Default analysis options.

        Analysis Options:
            combine_results (bool): If True flatten all component experiment
                results into a single ExperimentData container, including
                nested composite experiments. If False save each component
                experiment results as a separate child ExperimentData
                container (Default: False).
        """
        opts = super()._default_options()
        opts.combine_results = False
        return opts

    def set_options(self, **fields):
        super().set_options(**fields)
        # If combining results we need to recursively set combine_results
        # to True for any CompositeAnalysis component analysis because
        # there will be no saved children to attach those components
        # child results to.
        if fields.get("combine_results", False):
            for analysis in self._analyses:
                if isinstance(analysis, CompositeAnalysis):
                    analysis.options.combine_results = True

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

        if not self.options.combine_results:
            # Initialize child components if they are not initalized
            # This only needs to be done if results are not being combined
            self._add_child_data(experiment_data)

        # Run analysis with replace_results = True since we have already
        # created the copy if it was required
        return super().run(experiment_data, replace_results=True, **options)

    def _run_analysis(self, experiment_data: ExperimentData):
        # Return list of experiment data containers for each component experiment
        # containing the marginalied data from the composite experiment
        component_expdata = self._component_experiment_data(experiment_data)

        # Run the component analysis on each component data
        for i, sub_expdata in enumerate(component_expdata):
            # Since copy for replace result is handled at the parent level
            # we always run with replace result on component analysis
            self._analyses[i].run(sub_expdata, replace_results=True)

        # Analysis is running in parallel so we add loop to wait
        # for all component analysis to finish before returning
        # the parent experiment analysis results
        for sub_expdata in component_expdata:
            sub_expdata.block_for_results()

        # Optionally combine results from all component experiments
        # for adding to the main experiment data container
        if self.options.combine_results:
            return self._combine_results(component_expdata)

        return [], []

    def _component_experiment_data(self, experiment_data: ExperimentData) -> List[ExperimentData]:
        """Return a list of marginalized experiment data for component experiments.

        Args:
            experiment_data: a composite experiment experiment data container.

        Returns:
            The list of analysis-ready marginalized experiment data for each
            component experiment.

        Raises:
            AnalysisError: if the component experiment data cannot be extracted.
        """
        if not self.options.combine_results:
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
            sub_expdata._data.clear()
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
            for i, index in enumerate(metadata["composite_index"]):
                if index not in marginalized_data:
                    # Initialize data list for marginalized
                    marginalized_data[index] = []
                sub_data = {"metadata": metadata["composite_metadata"][i]}
                if "counts" in datum:
                    if composite_clbits is not None:
                        sub_data["counts"] = marginal_counts(datum["counts"], composite_clbits[i])
                    else:
                        sub_data["counts"] = datum["counts"]
                if "memory" in datum:
                    if composite_clbits is not None:
                        sub_data["memory"] = (
                            np.array(datum["memory"])[composite_clbits[i]]
                        ).tolist()
                    else:
                        sub_data["memory"] = datum["memory"]
                marginalized_data[index].append(sub_data)

        # Sort by index
        return [marginalized_data[i] for i in sorted(marginalized_data.keys())]

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
            subdata._type = experiment_types[i]
            subdata.metadata.update(component_metadata[i])

            if self.options.combine_results:
                # Explicitly set auto_save to false so the temporary
                # data can't accidentally be saved
                subdata.auto_save = False
            else:
                # Copy tags, share_level and auto_save from the parent
                # experiment data if results are not being combined.
                subdata.tags = experiment_data.tags
                subdata.share_level = experiment_data.share_level
                subdata.auto_save = experiment_data.auto_save

            component_expdata.append(subdata)

        return component_expdata

    def _combine_results(
        self, component_experiment_data: List[ExperimentData]
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
        if self.options.combine_results:
            # Optionally combine results into main container
            for i, sub_expdata in enumerate(component_experiment_data):
                figures += sub_expdata._figures.values()
                for result in sub_expdata.analysis_results():
                    # Add metadata to distinguish the component experiment
                    # the result was generated from
                    result.extra["component_experiment"] = {
                        "experiment_type": sub_expdata.experiment_type,
                        "component_index": i,
                    }
                    analysis_results.append(result)

        return analysis_results, figures
