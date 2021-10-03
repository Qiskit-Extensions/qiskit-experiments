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
T2Hahn Echo Experiment class.

"""

from typing import Union, Iterable, List, Optional

import numpy as np
from qiskit import QuantumCircuit, QiskitError
from qiskit.utils import apply_prefix
from qiskit.providers.options import Options
from qiskit.providers import Backend
from .t2hahn_analysis import T2HahnAnalysis

from qiskit_experiments.framework import BaseExperiment


class T2Hahn(BaseExperiment):
    r"""T2 Ramsey Experiment.

        # section: overview

            This experiment is used to estimate T2 noise of a single qubit.

            See `Qiskit Textbook <https://qiskit.org/textbook/ch-quantum-hardware/\
            calibrating-qubits-pulse.html>`_  for a more detailed explanation on
            these properties.

            This experiment consists of a series of circuits of the form

            .. parsed-literal::

                 ┌─────────┐┌──────────┐┌───────┐┌──────────┐┌──────────┐┌─┐
            q_0: ┤ RY(π/2) ├┤ DELAY(t) ├┤ RX(π) ├┤ DELAY(t) ├┤ RY(-π/2) ├┤M├
                 └─────────┘└──────────┘└───────┘└──────────┘└──────────┘└╥┘
            c: 1/═════════════════════════════════════════════════════════╩═
                                                                         0
            for each *t* from the specified delay times
            and the delays are specified by the user.
            The circuits are run on the device or on a simulator backend.

        # section: tutorial
            :doc:`/tutorials/t2hahn_characterization`

        """
    __analysis_class__ = T2HahnAnalysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments.
            unit (str): Unit of the delay times. Supported units are
                's', 'ms', 'us', 'ns', 'ps', 'dt'.
        """
        options = super()._default_experiment_options()

        options.delays = None
        options.unit = "s"

        return options

    def __init__(
        self,
        qubit: Union[int, Iterable[int]],
        delays: Union[List[float], np.array],
        unit: str = "s",
    ):
        """
        **T2 - Hahn Echo class**
        Initialize the T2 - Hahn Echo class
        Args:
            qubit: the qubit under test.
            delays (List[float)): delay times of the experiments.
            unit: Optional, time unit of `delays`.
                Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'. The unit is
                used for both T2Ramsey and for the frequency.

         Raises:
             QiskitError : Error for invalid input.
        """
        # Initialize base experiment
        super().__init__([qubit])
        # Set configurable options
        self.set_experiment_options(delays=delays, unit=unit)
        self._verify_parameters()

    def _verify_parameters(self):
        """
        Verify input correctness, raise QiskitError if needed.
        Args:
            qubit: the qubit under test.

        Raises:
            QiskitError : Error for invalid input.
        """
        if any(delay < 0 for delay in self.experiment_options.delays):
            raise QiskitError(
                f"The lengths list {self.experiment_options.delays} should only contain "
                "non-negative elements."
            )
        if len(set(self.experiment_options.delays)) != len(self.experiment_options.delays):
            raise QiskitError(
                f"The lengths list {self.experiment_options.delays} should not contain "
                "duplicate elements."
            )

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """
        Args:
            backend: Optional, a backend object.

        Returns:
            The experiment circuits.

        """
        conversion_factor = 1
        if self.experiment_options.unit == "dt":
            try:
                dt_factor = getattr(backend._configuration, "dt")
                conversion_factor = dt_factor
            except AttributeError as no_dt:
                raise AttributeError("Dt parameter is missing in backend configuration") from no_dt
        elif self.experiment_options.unit != "s":
            conversion_factor = apply_prefix(1, self.experiment_options.unit)

        circuits = []
        qubit = list(self._physical_qubits)
        for circ_index, delay in enumerate(self.experiment_options.delays):
            circ = QuantumCircuit(1, 1)
            # First Y rotation in 90 degrees
            circ.ry(np.pi / 2, qubit)  # Bring to qubits to X Axis
            circ.delay(delay, qubit, self.experiment_options.unit)
            circ.rx(np.pi, qubit)
            circ.delay(delay, qubit, self.experiment_options.unit)
            circ.ry(-np.pi / 2, qubit)  # Y90
            circ.measure(0, 0)  # measure
            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": delay,
                "unit": self.experiment_options.unit,
            }
            if self.experiment_options.unit == "dt":
                circ.metadata["dt_factor"] = dt_factor
            circuits.append(circ)

        return circuits
