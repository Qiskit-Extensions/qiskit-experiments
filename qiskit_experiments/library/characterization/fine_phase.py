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

"""Fine phase characterization experiment."""

from typing import Tuple, List, Optional

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.curve_analysis import ParameterRepr
from qiskit_experiments.curve_analysis.standard_analysis import ErrorAmplificationAnalysis


class FineZI(BaseExperiment):
    """Fine ZI phase characteriaztion.

    # section: overview

        This experiment is often used to characterize phase accumulation on the control qubit
        from the cross resonacne drive of the target qubit. This off-resonant drive may
        induce the Stark shfit, which appears as ZI term in the Hamiltonian.
        This term is usually eliminated in the echoed cross resonance sequence, however,
        this term has significant strength in non-echoed sequences such as DDCX [1].

        .. parsed-literal::

                 ┌─────────┐ ░ ┌─────┐┌─────┐   ┌─────┐┌─────┐ ░ ┌─────────┐┌─────────┐┌─┐
            q_0: ┤ Ry(π/2) ├─░─┤0    ├┤0    ├...┤0    ├┤0    ├─░─┤ Ry(π/2) ├┤ Rx(π/2) ├┤M├
                 ├─────────┤ ░ │  CR ││  CR │   │  CR ││  CR │ ░ └─────────┘└─────────┘└╥┘
            q_1: ┤ Ry(π/2) ├─░─┤1    ├┤1    ├...┤1    ├┤1    ├─░────────────────────────╫─
                 └─────────┘ ░ └─────┘└─────┘   └─────┘└─────┘ ░                        ║
            c: 1/═══════════════════════════════════════════════════════════════════════╩═
                                                                                        0

        Basically this experiment is designed based upon the similar idea to :class:`T2Ramsey`,
        in which the qubit to measure the phase accumulation is initialized in
        one of non-eigenbases of Pauli Z with Y90 pulse and projected back to the
        computational basis with another Y90 pulse after the amplification sequence.
        Note that here we assume small ZI term and apply :class:`ErrorAmplificationAnalysis`
        to the expeirment result. The X90 pulse in front of the measurement is to observe the
        state around equator of the Block sphere, which is typical requirement of
        amplification analysis, i.e. analysis of ping-pong pattern.

    # section: analysis_ref
        :py:class:`~qiskit_experiments.curve_analysis.ErrorAmplificationAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2008.08571

    """

    def __init__(
        self,
        qubits: Tuple,
        backend: Optional[Backend] = None,
    ):
        """Create new experiment.

        Args:
            qubits: Control and target qubit.
            backend: Optional, the backend to run the experiment on.
        """
        analysis = ErrorAmplificationAnalysis()
        analysis.set_options(
            fixed_parameters={
                "angle_per_gate": 0.0,
                "phase_offset": np.pi / 2,
                "amp": 1.0,
            },
            result_parameters=[ParameterRepr("d_theta", "zi_phase", "rad.")],
            outcome="1",
        )
        super().__init__(qubits, analysis=analysis, backend=backend)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that gate sequence is repeated.
            gate (Gate): This is the gate such as XGate() that will be in the circuits.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(20))
        options.gate = CXGate()

        return options

    def circuits(self) -> List[QuantumCircuit]:
        circuits = []
        for rep in self.experiment_options.repetitions:
            circuit = QuantumCircuit(2, 1)
            circuit.ry(np.pi/2, 0)
            circuit.ry(np.pi/2, 1)
            circuit.barrier()
            for _ in range(rep):
                circuit.append(self.experiment_options.gate, [0, 1])
            circuit.barrier()
            circuit.ry(np.pi/2, 0)
            circuit.rx(np.pi/2, 0)
            circuit.measure(0, 0)
            circuit.metadata = {
                "xval": rep
            }
            circuits.append(circuit)

        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
