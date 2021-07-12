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
Quantum Volume Experiment class.
"""

import copy
from typing import Union, Iterable, Optional, List
from numpy.random import Generator, default_rng
from qiskit.providers.backend import Backend
from qiskit.providers.options import Options

try:
    from qiskit import Aer

    HAS_SIMULATION_BACKEND = True
except ImportError:
    HAS_SIMULATION_BACKEND = False

from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume as QuantumVolumeCircuit
from qiskit import execute
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.experiment_data import ExperimentData
from .qv_analysis import QuantumVolumeAnalysis


class QuantumVolume(BaseExperiment):
    """Quantum Volume Experiment class

    Experiment Options:
        trials (int): Optional, number of times to generate new Quantum Volume circuits and
                    calculate their heavy output.
    """

    # Analysis class for experiment
    __analysis_class__ = QuantumVolumeAnalysis

    # ExperimentData class for the simulations
    __simulation_data__ = ExperimentData

    def __init__(
        self,
        qubits: Union[int, Iterable[int]],
        trials: Optional[int] = 100,
        seed: Optional[Union[int, Generator]] = None,
        simulation_backend: Optional[Backend] = None,
    ):
        """Quantum Volume experiment
        Args:
            qubits: the number of qubits or list of
                    physical qubits for the experiment.
            trials: number of trials to run the quantum volume circuit.
            seed: Seed or generator object for random number
                  generation. If None default_rng will be used.
            simulation_backend: the simulator backend to use to generate
                the expected results. the simulator must have a 'save_probabilities' method.
                if None Aer simulator will be used (in case Aer is not installed -
                qiskit.quantum_info will be used).
        """
        super().__init__(qubits)

        # Set configurable options
        self.set_experiment_options(trials=trials)

        # Set fixed options
        self._previous_trials = 0
        self._simulation_data = None

        if not isinstance(seed, Generator):
            self._rng = default_rng(seed=seed)
        else:
            self._rng = seed

        if not simulation_backend and HAS_SIMULATION_BACKEND:
            self._simulation_backend = Aer.get_backend("aer_simulator")
        else:
            self._simulation_backend = simulation_backend

    def _add_ideal_data(self, circuit, ideal_circuit, **run_options):
        if self._simulation_backend:
            ideal_result = execute(ideal_circuit, backend=self._simulation_backend, **run_options)
            probabilities = ideal_result.result().data().get("probabilities")
        else:
            from qiskit.quantum_info import Statevector

            state_vector = Statevector(ideal_circuit)
            probabilities = state_vector.probabilities()
        circuit.metadata["ideal_probabilities"] = probabilities

    @classmethod
    def _default_experiment_options(cls):
        return Options(trials=100)

    @property
    def simulation_data(self):
        """Return the ideal data of the experiment"""
        return self._simulation_data

    def _get_ideal_data(self, circuits, **run_options):
        """
        In case the user does not have Aer installed - use Terra to calculate the ideal state
        Args:
            circuits: the circuits to extract the ideal data from
        Returns:
            dict: data object with the circuit's metadata
                  and the probability for each state in each circuit
        """
        if self._simulation_backend:
            for circuit in circuits:
                circuit.save_probabilities()
            return execute(circuits, backend=self._simulation_backend, **run_options)
        else:
            from qiskit.quantum_info import Statevector

            sim_obj = []
            for circuit in circuits:
                state_vector = Statevector(circuit)
                sim_data = {
                    "probabilities": state_vector.probabilities(),
                    "metadata": circuit.metadata,
                }
                sim_obj.append(sim_data)
            return sim_obj

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Return a list of QV circuits, without the measurement instruction
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """
        circuits = []
        depth = self._num_qubits

        # Continue the trials numbers from previous experiments runs
        for trial in range(self._previous_trials + 1, self.experiment_options.trials + 1):
            qv_circ = QuantumVolumeCircuit(depth, depth, seed=self._rng)
            qv_circ.measure_active()
            qv_circ.metadata = {
                "experiment_type": self._type,
                "depth": depth,
                "trial": trial,
                "qubits": self.physical_qubits,
            }
            ideal_circuit = qv_circ.remove_final_measurements(inplace=False)
            ideal_circuit.save_probabilities()
            self._add_ideal_data(qv_circ, ideal_circuit)
            circuits.append(qv_circ)


        return circuits
