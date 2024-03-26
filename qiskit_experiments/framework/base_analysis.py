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
Base analysis class.
"""
from abc import ABC, abstractmethod
import copy
from collections import OrderedDict
from datetime import datetime
from typing import List, Tuple, Union, Dict
import warnings

from dateutil import tz

from qiskit_experiments.database_service.device_component import Qubit
from qiskit_experiments.framework import Options
from qiskit_experiments.framework.containers.artifact_data import ArtifactData
from qiskit_experiments.framework.store_init_args import StoreInitArgs
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.framework.containers import FigureData, FigureType
from qiskit_experiments.framework.configs import AnalysisConfig
from qiskit_experiments.framework.analysis_result_data import AnalysisResultData, as_table_element


class BaseAnalysis(ABC, StoreInitArgs):
    """Abstract base class for analyzing Experiment data.

    The data produced by experiments (i.e. subclasses of BaseExperiment)
    are analyzed with subclasses of BaseAnalysis. The analysis is
    typically run after the data has been gathered by the experiment.
    For example, an analysis may perform some data processing of the
    measured data and a fit to a function to extract a parameter.

    Analysis subclasses must implement the abstract method `_run_analysis`.
    This method should not have side-effects on the analysis class itself
    since it could potentially be called asynchronously in multiple threads.
    Any configurable option values should be specified in the `_default_options`
    class method. These values can be overriden by a user by calling the
    `set_options` method or for a single-run can be specified by passing kwarg
    options to the :meth:`run` method.
    """

    def __init__(self):
        """Initialize the analysis object."""
        # Analysis options
        self._options = self._default_options()

        # Store keys of non-default options
        self._set_options = set()

    def config(self) -> AnalysisConfig:
        """Return the config dataclass for this analysis"""
        args = tuple(getattr(self, "__init_args__", OrderedDict()).values())
        kwargs = dict(getattr(self, "__init_kwargs__", OrderedDict()))
        # Only store non-default valued options
        options = dict((key, getattr(self._options, key)) for key in self._set_options)
        return AnalysisConfig(
            cls=type(self),
            args=args,
            kwargs=kwargs,
            options=options,
        )

    @classmethod
    def from_config(cls, config: Union[AnalysisConfig, Dict]) -> "BaseAnalysis":
        """Initialize an analysis class from analysis config"""
        if isinstance(config, dict):
            config = AnalysisConfig(**config)
        ret = cls(*config.args, **config.kwargs)
        if config.options:
            ret.set_options(**config.options)
        return ret

    def copy(self) -> "BaseAnalysis":
        """Return a copy of the analysis"""
        # We want to avoid a deep copy be default for performance so we
        # need to also copy the Options structures so that if they are
        # updated on the copy they don't effect the original.
        ret = copy.copy(self)
        ret._options = copy.copy(self._options)
        ret._set_options = copy.copy(self._set_options)
        return ret

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options common to all analyses.

        Analysis Options:
            figure_names (str or List[str]): Identifier of figures that appear in the
                experiment data to sort figures by name.
        """
        options = Options()
        # figure names can be set for each analysis by calling
        # experiment_obj.analysis.set_options(figure_names=FIGURE_NAMES)
        options.figure_names = None
        return options

    @property
    def options(self) -> Options:
        """Return the analysis options for :meth:`run` method."""
        return self._options

    def set_options(self, **fields):
        """Set the analysis options for :meth:`run` method.

        Args:
            fields: The fields to update the options
        """
        self._options.update_options(**fields)
        self._set_options = self._set_options.union(fields)

    def run(
        self,
        experiment_data: ExperimentData,
        replace_results: bool = False,
        **options,
    ) -> ExperimentData:
        """Run analysis and update ExperimentData with analysis result.

        Args:
            experiment_data: the experiment data to analyze.
            replace_results: If True clear any existing analysis results, figures,
                             and artifacts in the experiment data and replace with
                             new results. See note for additional information.
            options: additional analysis options. See class documentation for
                     supported options.

        Returns:
            An experiment data object containing analysis results, figures, and artifacts.

        Raises:
            QiskitError: If experiment_data container is not valid for analysis.

        .. note::
            **Updating Results**

            If analysis is run with ``replace_results=True`` then any analysis results,
            figures, and artifacts in the experiment data will be cleared and replaced with the
            new analysis results. Saving this experiment data will replace any
            previously saved data in a database service using the same experiment ID.

            If analysis is run with ``replace_results=False`` and the experiment data
            being analyzed has already been saved to a database service, or already
            contains analysis results or figures, a copy with a unique experiment ID
            will be returned containing only the new analysis results and figures.
            This data can then be saved as its own experiment to a database service.
        """
        # Make a new copy of experiment data if not updating results
        if not replace_results and _requires_copy(experiment_data):
            experiment_data = experiment_data.copy()

        # Set Analysis options
        if not options:
            analysis = self
        else:
            analysis = self.copy()
            analysis.set_options(**options)

        def run_analysis(expdata: ExperimentData):
            # Clearing previous analysis data
            experiment_data._clear_results()

            if not expdata.data():
                warnings.warn("ExperimentData object data is empty.\n")

            # Making new analysis
            results, figures = analysis._run_analysis(expdata)

            if results:
                for result in results:
                    if isinstance(result, AnalysisResultData):
                        # Populate missing data fields
                        if not result.experiment_id:
                            result.experiment_id = expdata.experiment_id
                        if not result.experiment:
                            result.experiment = expdata.experiment_type
                        if not result.device_components:
                            result.device_components = analysis._get_experiment_components(expdata)
                        if not result.backend:
                            result.backend = expdata.backend_name
                        if not result.created_time:
                            result.created_time = datetime.now(tz.tzlocal())
                        if not result.run_time:
                            result.run_time = expdata.running_time

                        # To canonical kwargs to add to the analysis table.
                        table_format = as_table_element(result)

                        # Remove result_id to make sure the id is unique in the scope of the container.
                        # This will let the container generate a unique id.
                        del table_format["result_id"]

                        expdata.add_analysis_results(**table_format)
                    elif isinstance(result, ArtifactData):
                        if not result.experiment_id:
                            result.experiment_id = expdata.experiment_id
                        if not result.device_components:
                            result.device_components = analysis._get_experiment_components(expdata)
                        if not result.experiment:
                            result.experiment = expdata.experiment_type
                        expdata.add_artifacts(result)
                    else:
                        raise TypeError(
                            f"Invalid object type {result.__class__.__name__} for analysis results. "
                            "This data cannot be stored in the experiment data."
                        )

            if figures:
                figure_to_add = []
                for figure in figures:
                    if not isinstance(figure, FigureData):
                        qubits_repr = "_".join(
                            map(str, expdata.metadata.get("device_components", [])[:5])
                        )
                        short_id = expdata.experiment_id[:8]
                        figure = FigureData(
                            figure=figure,
                            name=f"{expdata.experiment_type}_{qubits_repr}_{short_id}.svg",
                        )
                    figure_to_add.append(figure)
                expdata.add_figures(figure_to_add, figure_names=analysis.options.figure_names)

        experiment_data.add_analysis_callback(run_analysis)

        return experiment_data

    def _get_experiment_components(self, experiment_data: ExperimentData):
        """Subclasses may override this method to specify the experiment components."""
        if "physical_qubits" in experiment_data.metadata:
            experiment_components = [
                Qubit(qubit) for qubit in experiment_data.metadata["physical_qubits"]
            ]
        else:
            experiment_components = []

        return experiment_components

    @abstractmethod
    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[Union[AnalysisResultData, ArtifactData]], List[FigureType]]:
        """Run analysis on circuit data.

        Args:
            experiment_data: the experiment data to analyze.

        Returns:
            A pair ``(analysis_results, figures)`` where ``analysis_results``
            is a list of :class:`.AnalysisResultData` objects, and ``figures``
            is a list of any figures for the experiment.

        Raises:
            AnalysisError: If the analysis fails.
        """
        # NOTE: passing kwarg options to _run_analysis should be removed once
        pass

    def __json_encode__(self):
        return self.config()

    @classmethod
    def __json_decode__(cls, value):
        return cls.from_config(value)


def _requires_copy(experiment_data) -> bool:
    """Return True if a copy of the experiment data should be made."""
    # If data is from DB or contains analysis results it should be copied
    if (
        experiment_data._created_in_db
        or experiment_data._analysis_results
        or experiment_data._figures
    ):
        return True

    # Check child data:
    if hasattr(experiment_data, "_child_data"):
        for subdata in experiment_data._child_data.values():
            if _requires_copy(subdata):
                return True

    # No Copy required
    return False
