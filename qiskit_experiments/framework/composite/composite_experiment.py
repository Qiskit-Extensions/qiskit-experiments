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
Composite Experiment abstract base class.
"""

import warnings
from abc import abstractmethod
from typing import List, Optional

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend, BaseJob

from qiskit_experiments.framework import BaseExperiment, ExperimentData
from .composite_analysis import CompositeAnalysis
from .composite_experiment_data import CompositeExperimentData


class CompositeExperiment(BaseExperiment):
    """Composite Experiment base class.

    Note:
        Composite experiment defines different option handling policies for each
        type of options.

        * transpile options: The transpile options set to each nested experiment are retained.
          Thus, the experiment can transpile experimental circuits individually and combine.
          Note that no transpile option can be set to composite experiment itself.

        * experiment options: Same with transpile options.

        * analysis options: Same with transpile options. However, one can set analysis options
          to the composite experiment. The set value will override all analysis configurations
          of the nested experiments.

        * run options: The run options set to nested experiments are discarded.
          This is because Qiskit doesn't assume a backend that can execute each circuit
          with different run options in a single job. If you want to keep run options
          set to the individual experiment, you need to individually run these experiments.

    """

    __analysis_class__ = CompositeAnalysis
    __experiment_data__ = CompositeExperimentData

    def __init__(self, experiments, qubits, experiment_type=None):
        """Initialize the composite experiment object.

        Args:
            experiments (List[BaseExperiment]): a list of experiment objects.
            qubits (int or Iterable[int]): the number of qubits or list of
                                           physical qubits for the experiment.
            experiment_type (str): Optional, composite experiment subclass name.
        """
        self._experiments = experiments
        self._num_experiments = len(experiments)
        super().__init__(qubits, experiment_type=experiment_type)

    def run_transpile(self, backend: Backend, **options) -> List[QuantumCircuit]:
        """Run transpile and returns transpiled circuits.

        Args:
            backend: Target backend.
            options: User provided runtime options.

        Returns:
            Transpiled circuit to execute.
        """
        # Generate a set of transpiled circuits for each nested experiment and have them as a list.
        # In each list element, a list of quantum circuit for nested experiment is stored.

        # : List[List[QuantumCircuit]]
        experiment_circuits_list = list(
            map(lambda expr: expr.run_transpile(backend), self._experiments)
        )

        # This is not identical to the `num_qubits` when the backend is AerSimulator.
        # In this case, usually a circuit qubit number is determined by the maximum qubit index.
        n_qubits = 0
        for circuits in experiment_circuits_list:
            circuit_qubits = [circuit.num_qubits for circuit in circuits]
            n_qubits = max(n_qubits, *circuit_qubits)

        # merge circuits
        return self._flatten_circuits(experiment_circuits_list, n_qubits)

    @abstractmethod
    def _flatten_circuits(
        self,
        circuits: List[List[QuantumCircuit]],
        num_qubits: int,
    ) -> List[QuantumCircuit]:
        """An abstract method to control flattening logic of sub experiments.

        This method takes a nested list of circuits, which corresponds to a
        list of circuits generated by each experiment, and generates a single list of circuits
        that is executed on the backend. This flattening logic may depend on the
        type of composite experiment.
        """
        pass

    def circuits(self, backend: Optional[Backend] = None):
        """Composite experiment does not provide this method.

        Args:
            backend: The targe backend.

        Raises:
            QiskitError: When this method is called.
        """
        raise QiskitError(
            f"{self.__class__.__name__} does not generate experimental circuits by itself. "
            "Call the corresponding method of individual experiment class to find circuits, "
            "or call `run_transpile` method to get circuits run on the target backend."
        )

    @property
    def num_experiments(self):
        """Return the number of sub experiments"""
        return self._num_experiments

    def component_experiment(self, index=None):
        """Return the component Experiment object.
        Args:
            index (int): Experiment index, or ``None`` if all experiments are to be returned.
        Returns:
            BaseExperiment: The component experiment(s).
        """
        if index is None:
            return self._experiments
        return self._experiments[index]

    def component_analysis(self, index):
        """Return the component experiment Analysis object"""
        return self.component_experiment(index).analysis()

    def _add_job_metadata(self, experiment_data, job, **run_options):
        # Add composite metadata
        super()._add_job_metadata(experiment_data, job, **run_options)

        # Add sub-experiment options
        for i in range(self.num_experiments):
            sub_exp = self.component_experiment(i)

            # Run and transpile options are always overridden
            if (
                sub_exp.run_options != sub_exp._default_run_options()
                or sub_exp.transpile_options != sub_exp._default_transpile_options()
            ):

                warnings.warn(
                    "Sub-experiment run and transpile options"
                    " are overridden by composite experiment options."
                )
            sub_data = experiment_data.component_experiment_data(i)
            sub_exp._add_job_metadata(sub_data, job, **run_options)

    def set_transpile_options(self, **fields):
        """Composite experiment itself doesn't provide transpile options."""
        warnings.warn(
            "A composite experiment class doesn't provide transpile options. "
            "Note that transpile options are provided by each nested experiment, "
            f"and thus provided options here {fields} are just discarded.",
            UserWarning,
        )
        super().set_transpile_options(**fields)
