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
T1 Experiment class.
"""

from typing import List, Optional, Union
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend
from qiskit.test.mock import FakeBackend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis.t1_analysis import T1Analysis


class T1(BaseExperiment):
    r"""
    T1 experiment class

    # section: overview

        Design and analyze experiments for estimating T\ :sub:`1` relaxation time of the qubit.

        Each experiment consists of the following steps:

        1. Circuits generation: the circuits set the qubit in the excited state,
        wait different time intervals, then measure the qubit.

        2. Backend execution: actually running the circuits on the device
        (or simulator).

        3. Analysis of results: deduction of T\ :sub:`1`\ , based on the outcomes,
        by fitting to an exponential curve.

    """

    __analysis_class__ = T1Analysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments in seconds.
        """
        options = super()._default_experiment_options()
        options.delays = None
        return options

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the T1 experiment class

        Args:
            qubit: the qubit whose T1 is to be estimated
            delays: delay times of the experiments in seconds
            backend: Optional, the backend to run the experiment on.

        Raises:
            ValueError: if the number of delays is smaller than 3
        """
        # Initialize base experiment
        super().__init__([qubit], backend=backend)

        # Set experiment options
        self.set_experiment_options(delays=delays)

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)

        # Scheduling parameters
        if not self._backend.configuration().simulator and not isinstance(backend, FakeBackend):
            timing_constraints = getattr(self.transpile_options, "timing_constraints", {})
            if "acquire_alignment" not in timing_constraints:
                timing_constraints["acquire_alignment"] = 16
            scheduling_method = getattr(self.transpile_options, "scheduling_method", "alap")
            self.set_transpile_options(
                timing_constraints=timing_constraints, scheduling_method=scheduling_method
            )

    def circuits(self) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Returns:
            The experiment circuits
        """
        circuits = []
        for delay in self.experiment_options.delays:
            circ = QuantumCircuit(1, 1)
            circ.x(0)
            circ.barrier(0)
            circ.delay(delay, 0, "s")
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": delay,
                "unit": "s",
            }

            circuits.append(circ)

        if self.backend and hasattr(self.backend.configuration(), "dt"):
            transpiled_circuits = transpile(
                circuits, self.backend, **self.transpile_options.__dict__
            )
            for circ, tcirc in zip(circuits, transpiled_circuits):
                for op, _, _ in tcirc.data:
                    if op.name == "delay":
                        circ.metadata["xval"] = op.params[0] * self.backend.configuration().dt
                        break

        return circuits
