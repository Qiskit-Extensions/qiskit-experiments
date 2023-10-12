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

"""A class to run several experiments one after the other."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from qiskit import QuantumCircuit
from qiskit.providers import Backend, Job
from qiskit.providers.options import Options

from qiskit_experiments.framework.base_experiment import (
    BaseExperiment,
    ExperimentData,
    BaseAnalysis,
)
from qiskit_experiments.framework.composite.composite_experiment import CompositeExperiment
from qiskit_experiments.exceptions import AnalysisError


class BaseTransitionCallable(ABC):
    """A class that holds a method to determine how to transition between experiments.

    Developers can create arbitrary logic to transition between experiments based on the
    result of the previously run experiments. To do this the :meth:`__call__` method
    must be implemented. This method return an int that determines by how many
    sub-experiments we move in the chain of experiments. For example, if the returned
    value is 0 we repeat the current experiment and if it is 1 we run the next experiment
    in the chain.
    """

    @abstractmethod
    def __call__(
        self,
        experiment_data: ExperimentData,
        **kwargs,
    ) -> int:
        """The call method determines if the ChainedExperiment can transition to the next experiment.

        Args:
            experiment_data: The experiment data. Typically, the children will contain the
                experiment data of the sub-experiments.
            kwargs: Additional keyword arguments.

        Returns:
            An integer that tells the :class:`ChainedExperiment` by how much it can increment
            the experiment index.
        """

    def __eq__(self, other):
        """Check equality, allows for serialization testing."""
        return isinstance(self, type(other))


class GoodExperimentTransition(BaseTransitionCallable):
    """An experiment transition callable that transitions on good results.

    If the quality of the previous experiment is ``good`` then we transition to the next
    experiment, i.e. return 1 otherwise we return 0, i.e. repeat the current experiment.
    """

    def __call__(self, experiment_data: ExperimentData, **kwargs) -> int:
        """A simple experiment transition callback based on the quality of the last result."""

        if experiment_data.child_data(-1).analysis_results(0).quality == "good":
            return 1
        else:
            return 0


class AlwaysTransition(BaseTransitionCallable):
    """A simple experiment transition callable that always transitions to the next experiment."""

    def __call__(self, experiment_data: ExperimentData, **kwargs) -> int:
        """An experiment transition callback that always moves to the next experiment."""
        return 1


class ChainedExperiment(CompositeExperiment):
    """An experiment made of several sub-experiments ran one after another.

    The chained experiment works using analysis callback functions and an index to the experiment
    in the chain of experiments that needs to be run. Each time an experiment is run there are
    two analysis callback functions that are added to the experiment data as follows.

    1. An experiment transition callback. This is a function that determines relative changes to
    the experiment index. For example, if the next experiment in the chain is to be executed then
    this function returns 1. However, if the current experiment is to be repeated then this
    function will return 0.

    2. A callback to launch the next experiment based on the value of the experiment index.

    The experiment data from this class will contain all the sub-experiments as children.
    """

    def __init__(
        self,
        experiments: List[BaseExperiment],
        transition_callback: Optional[BaseTransitionCallable] = None,
    ):
        """Setup the chained experiment.

        Args:
            experiments: The list of experiments to run.
            transition_callback: The method that determines which experiment to run next.
        """
        qubits, backend = set(), None
        for exp in experiments:
            qubits.update(exp.physical_qubits)

            if backend is None and exp.backend:
                backend = exp.backend

        # The index that points to the experiment that is running.
        self._current_index = 0

        # A counter to safe-guard against endless experiment runs.
        self._number_of_runs = 0

        super().__init__(experiments, list(qubits), backend=backend)

        # The arguments for the transition callback function.
        self._transition_options = self._default_transition_options()

        if transition_callback:
            self.set_experiment_options(callback=transition_callback)

    def current_index(self) -> int:
        """Returns the index of the experiment that is currently being run in the chain."""
        return self._current_index

    def _set_backend(self, backend: Backend):
        """Set the backend of the experiment at the current pointer index."""
        self._backend = backend

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """The default options for this experiment.

        Experiment Options:
            callback: The function that determines how to transition between experiments
                in the chain. By default this is a callback that increments the experiment
                if the quality of the result is good.
            max_runs: The maximum number of times that the run method of the experiments
                can be called. By default this value is set to 100.
        """
        options = super()._default_experiment_options()
        options.callback = GoodExperimentTransition()
        options.max_runs = 100
        return options

    @classmethod
    def _default_transition_options(cls) -> Options:
        """Default options to control how transitions between experiments occur."""
        return Options()

    @property
    def transition_options(self) -> Options:
        """Options that can be given to the transition callback."""
        return self._transition_options

    def set_transition_options(self, **fields):
        """Update the options of the transition callable."""
        self._transition_options.update_options(**fields)

    def run(
        self,
        backend: Optional[Backend] = None,
        analysis: Optional[Union[BaseAnalysis, None]] = "default",
        timeout: Optional[float] = None,
        **run_options,
    ) -> ExperimentData:
        """Run the chained experiment by transitioning between sub-experiments with the index."""
        experiment_data = super().run(
            backend=backend, analysis=None, timeout=timeout, **run_options
        )

        return self._run_index(experiment_data)

    def _finalize(self):
        """Nothing to do as run options can differ from one experiment to the next."""
        pass

    def _run_jobs(self, circuits: List[QuantumCircuit], **run_options) -> List[Job]:
        """Do not run anything yet."""
        return []

    def _run_index(self, experiment_data: ExperimentData) -> ExperimentData:
        """Recursively run the experiments.

        The experiment at the current index is run and then the index is updated using an analysis
        callback.

        Args:
            experiment_data: The experiment data to which the experiment data of the
                run experiment will be added as a child.

        Returns:
            The experiment data with an extra child experiment data added to it.

        Raises:
            AnalysisError: if the maximum number of experiment executions has been exceeded.
        """

        # Termination condition.
        if self._current_index >= self.num_experiments:
            return experiment_data

        if self._number_of_runs > self.experiment_options.max_runs:
            raise AnalysisError("The maximum allowed number of runs has been exceeded.")

        self.backend = self.component_experiment(self._current_index).backend

        exp_data = self.component_experiment(self._current_index).run()
        experiment_data.add_child_data(exp_data)
        self._number_of_runs += 1

        # Add the transition callback as an analysis callback to the experiment data.
        experiment_data.add_analysis_callback(self._transition_callback)

        # recursion
        experiment_data.add_analysis_callback(self._run_index)

        return experiment_data

    def _transition_callback(self, experiment_data):
        """Update the index pointing to the experiment to run."""
        self._current_index += self.experiment_options.callback(
            experiment_data,
            **self._transition_options.__dict__,
        )

    def circuits(self) -> List[QuantumCircuit]:
        """Returns the circuits of the current experiment."""
        return self.component_experiment(self._current_index).circuits()
