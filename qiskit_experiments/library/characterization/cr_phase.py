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

"""Cross resonance Hamiltonian phase characterization."""

from typing import Iterable, List, Optional, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter, Gate
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.curve_analysis import OscillationAnalysis


class CrossResonancePhase(BaseExperiment):
    """An experiment to characterize phase errors in cross-resonance gates.

    # section: overview

        This experiment scans the phase of the cross-resonance drive to find the
        value that most closely produces a ZX-like drive. The following circuits
        are executed

        .. parsed-literal::

                 ┌───┐┌─────────┐┌───┐┌─────────┐
            q_0: ┤ X ├┤0        ├┤ X ├┤0        ├───
                 └───┘│  crp(φ) │└───┘│  crm(φ) │┌─┐
            q_1: ─────┤1        ├─────┤1        ├┤M├
                      └─────────┘     └─────────┘└╥┘
            c: 1/═════════════════════════════════╩═
                                                  0

        Here, the phase φ is scanned over a range. The resulting oscillation is fit
        to a cosine function.

    """

    def __init__(
        self,
        qubits: Tuple[int, int],
        crp: ScheduleBlock,
        crm: ScheduleBlock,
        backend: Optional[Backend] = None,
        phases: Optional[Iterable] = None,
    ):
        """Create a new experiment to scan the phase of the cross-resonance gate.

        Args:
            qubits: The physical qubits on which to run as (control, target).
            crp: The positive cross-resonance schedule with a parameterized amplitude
                with the form :math:`A\cdot e^{-i\phi}` where :math:`A` is a real number
                and :math:`\phi` is a Qiskit :class:`Parameter`.
            crm: The negative cross-resonance schedule with a parameterized amplitude
                with the form :math:`-A\cdot e^{-i\phi}` where :math:`A` is a real number,
                identical to the a in ``crp`` and :math:`\phi` is a Qiskit :class:`Parameter`.
            backend: The backend on which to run the experiment.
            phases: The phases of the CR drives that will be scanned.
        """
        super().__init__(qubits, OscillationAnalysis(), backend)
        self.set_experiment_options(crp=crp, crm=crm)
        if phases is not None:
            self.set_experiment_options(phases=phases)
        
        self.analysis.set_options(outcome="1")

    def _default_experiment_options(cls) -> Options:
        """The default experiment options.

        Experiment Options:
            phases: The sequence of phases that will be run.
        """
        options = super()._default_experiment_options()
        options.phases = np.linspace(-np.pi/2, np.pi/2, 61)
        options.crp = None
        options.crm = None
        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpile options.

        The basis gates option should not be changed since it will affect the gates and
        the pulses that are run on the hardware.
        """
        options = super()._default_transpile_options()
        options.basis_gates = ["crm", "crp", "x", "rz"]
        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return the circuits of the experiment."""

        param = tuple(self.experiment_options.crp.parameters)[0]

        circuit = QuantumCircuit(2, 1)

        circuit.x(0)
        circuit.append(Gate(name="crp", num_qubits=2, params=[param]), (0, 1))
        circuit.x(0)
        circuit.append(Gate(name="crm", num_qubits=2, params=[param]), (0, 1))
        circuit.measure(1, 0)
        circuit.add_calibration("crp", self.physical_qubits, self.experiment_options.crp, [param])
        circuit.add_calibration("crm", self.physical_qubits, self.experiment_options.crm, [param])

        circuits = []
        for phase in self.experiment_options.phases:
            phase = np.round(phase, 6)
            assigned_circ = circuit.assign_parameters({param: phase}, inplace=False)
            assigned_circ.metadata = {
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "xval": phase,
            }

            circuits.append(assigned_circ)

        return circuits
