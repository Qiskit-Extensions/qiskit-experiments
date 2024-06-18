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

"""Iterated Experiment class."""

from typing import Callable, List, Optional, Union

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.providers.options import Options

from qiskit_experiments.framework.base_experiment import BaseExperiment
from qiskit_experiments.framework.base_analysis import BaseAnalysis
from qiskit_experiments.framework.experiment_data import ExperimentData


class IteratedExperiment(BaseExperiment):
    """Iterated experiments class.

    This experiment is initialized from another experiment and will iterate that experiment
    until a condition specified by a callback function is reached. Iterated experiments
    make most sense for error amplifying calibration experiments where the parameter of
    a pulse is updated until a given threshold is reached. The code below is an example
    of how to use this class to run an iterative fine amplitude calibration experiment

    .. code::

        amp_cal = FineXAmplitudeCal(0, cals, "x", backend=backend)
        amp_iter = IteratedExperiment(amp_cal)
        exp_data = iter_exp.run()

    Note: the transpile options of this class are irrelevant. The options of the experiment
    being iterated are set by accessing the experiment directly, e.g.

    .. code::

        amp_iter.experiment.set_transpile_options(**options_to_set)

    The run options of :class:`IteratedExperiment` control the behaviour of the iterated
    experiment such as the maximum number of iterations and can be used to influence
    the stopping condition by passing them to the callback function used to determine
    if another iteration is needed.
    """

    def __init__(self, experiment: BaseExperiment, callback: Optional[Callable] = None):
        """Initialize the iterated experiment.

        Args:
            experiment: The experiment to iterate.
            callback: A callback function to determine if the experiment should keep iterating.
                This callback function is called at each iteration and must have the signature
                :code:`def callback(exp_data: ExperimentData, run_options: Options) -> bool`.
        """
        super().__init__(experiment.physical_qubits)

        self._experiment = experiment
        self._callback = callback or self.tolerance
        self._iter_count = None

    @property
    def experiment(self) -> BaseExperiment:
        """Return the experiment that is iterated."""
        return self._experiment

    @classmethod
    def _default_run_options(cls):
        """Default options for an iterated experiment."""
        options = Options(n_iter=10, tol=0.001)

        return options

    def run(
        self,
        backend: Optional[Backend] = None,
        analysis: Optional[Union[BaseAnalysis, None]] = "default",
        timeout: Optional[float] = None,
        **run_options,
    ) -> ExperimentData:
        """Run the iterated experiment.

        The iterated experiment is run by iteratively running the sub-experiment. This experiment
        is run until a stopping condition is met.

        Returns:
            An instance of :class:`ExperimentData` that contains links to all the child
            experiment data obtained while running the experiment.
        """
        self._iter_count = 0

        exp_data = None
        all_exp_data = ExperimentData(experiment=self)

        while self._keep_running(exp_data):
            exp_data = self._experiment.run(
                backend=backend, analysis=analysis, timeout=timeout, **run_options
            )

            exp_data.block_for_results()

            self._iter_count += 1

            all_exp_data.add_child_data(exp_data)

        return all_exp_data

    def _keep_running(self, exp_data: ExperimentData) -> bool:
        """This function determines if the iterated experiment should keep running.

        Args:
            exp_data: The experiment data that this method will use in conjunction with the
                run options tto determine if the experiment must be run again.

        Returns:
            True if another iteration should be run and False if the experiment should not
            continue iterating.
        """

        # First guard against running an infinite number of times.
        if self._iter_count > self.run_options.n_iter:
            return False

        # Edge case where the experiment has not been run yet.
        if exp_data is None:
            return True

        # Use the callback to determine if the halting condition is reached.
        return self._callback(exp_data, self.run_options)

    def circuits(self) -> List[QuantumCircuit]:
        """Return the circuits of the experiment being iterated."""
        return self._experiment.circuits()

    @staticmethod
    def tolerance(exp_data: ExperimentData, options: Options):
        """Return true if the last value is below the tolerance."""

        if exp_data.analysis_results(-1).value.value < options.tol:
            return False

        return True
