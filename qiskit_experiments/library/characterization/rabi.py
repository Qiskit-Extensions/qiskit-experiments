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

"""Rabi amplitude experiment."""

from typing import Iterable, List, Optional, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.exceptions import QiskitError

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.curve_analysis import ParameterRepr, OscillationAnalysis


class Rabi(BaseExperiment):
    """An experiment that scans a pulse amplitude to calibrate rotations between 0 and 1.

    # section: overview

        The circuits have a custom rabi gate with the pulse schedule attached to it
        through the calibrations. The circuits are of the form:

        .. parsed-literal::

                       ┌───────────┐ ░ ┌─┐
                  q_0: ┤ Rabi(amp) ├─░─┤M├
                       └───────────┘ ░ └╥┘
            measure: 1/═════════════════╩═
                                        0

        The user provides his own schedule for the Rabi at initialization which must have one
        free parameter, i.e. the amplitude to scan and a drive channel which matches the qubit.

    # section: tutorial
        :doc:`/tutorials/calibrating_armonk`

        See also `Qiskit Textbook <https://qiskit.org/textbook/ch-quantum-hardware/\
        calibrating-qubits-pulse.html>`_
        for the pulse level programming of a Rabi experiment.

    """

    __analysis_class__ = OscillationAnalysis
    __gate_name__ = "Rabi"

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default option values for the experiment :meth:`run` method."""
        options = super()._default_run_options()

        options.meas_level = MeasLevel.KERNELED
        options.meas_return = "single"

        return options

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the pulse if no schedule is given.

        Experiment Options:
            amplitudes (iterable): The list of amplitude values to scan.
            schedule (ScheduleBlock): The schedule for the Rabi pulse. This schedule must have
                exactly one free parameter. The drive channel should match the qubit.

        """
        options = super()._default_experiment_options()

        options.amplitudes = np.linspace(-0.95, 0.95, 51)
        options.schedule = None

        return options

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.result_parameters = [ParameterRepr("freq", "rabi_rate")]
        options.xlabel = "Amplitude"
        options.ylabel = "Signal (arb. units)"
        options.normalization = True

        return options

    def __init__(
        self,
        qubit: int,
        schedule: ScheduleBlock,
        amplitudes: Optional[Iterable[float]] = None,
        backend: Optional[Backend] = None,
    ):
        """Initialize a Rabi experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the Rabi experiment.
            schedule: The schedule that will be used in the Rabi experiment. This schedule
                should have one free parameter namely the amplitude.
            amplitudes: The pulse amplitudes that one wishes to scan. If this variable is not
                specified it will default to :code:`np.linspace(-0.95, 0.95, 51)`.
            backend: Optional, the backend to run the experiment on.
        """
        super().__init__([qubit], backend=backend)

        if amplitudes is not None:
            self.experiment_options.amplitudes = amplitudes

        self.experiment_options.schedule = schedule

    def _pre_circuit(self) -> QuantumCircuit:
        """A circuit with operations to perform before the Rabi."""
        return QuantumCircuit(1)

    def _template_circuit(self) -> Tuple[QuantumCircuit, Parameter]:
        """Return the template quantum circuit."""
        sched = self.experiment_options.schedule
        param = next(iter(sched.parameters))

        if len(sched.parameters) != 1:
            raise QiskitError(
                f"Schedule {sched} for {self.__class__.__name__} experiment must have "
                f"exactly one free parameter, found {sched.parameters} parameters."
            )

        gate = Gate(name=self.__gate_name__, num_qubits=1, params=[param])

        circuit = self._pre_circuit()
        circuit.append(gate, (0,))
        circuit.measure_active()
        circuit.add_calibration(gate, self._physical_qubits, sched, params=[param])

        return circuit, param

    def circuits(self) -> List[QuantumCircuit]:
        """Create the circuits for the Rabi experiment.

        Returns:
            A list of circuits with a rabi gate with an attached schedule. Each schedule
            will have a different value of the scanned amplitude.
        """

        # Create template circuit
        circuit, param = self._template_circuit()

        # Create the circuits to run
        circs = []
        for amp in self.experiment_options.amplitudes:
            amp = np.round(amp, decimals=6)
            assigned_circ = circuit.assign_parameters({param: amp}, inplace=False)
            assigned_circ.metadata = {
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "xval": amp,
                "unit": "arb. unit",
                "amplitude": amp,
            }

            circs.append(assigned_circ)

        return circs


class EFRabi(Rabi):
    """An experiment that scans the amplitude of a pulse inducing rotations between 1 and 2.

    # section: overview

        This experiment is a subclass of the :class:`Rabi` experiment but takes place between
        the first and second excited state. An initial X gate populates the first excited state.
        The Rabi pulse is applied on the 1 <-> 2 transition (sometimes also labeled the e <-> f
        transition). The necessary frequency shift (typically the qubit anharmonicity) is given
        through the pulse schedule given at initialization. The schedule is then also stored in
        the experiment options. The circuits are of the form:

        .. parsed-literal::

                       ┌───┐┌───────────┐ ░ ┌─┐
                  q_0: ┤ X ├┤ Rabi(amp) ├─░─┤M├
                       └───┘└───────────┘ ░ └╥┘
            measure: 1/══════════════════════╩═
                                             0

    """

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.result_parameters = [ParameterRepr("freq", "rabi_rate_12")]

        return options

    def _pre_circuit(self) -> QuantumCircuit:
        """A circuit with operations to perform before the Rabi."""
        circ = QuantumCircuit(1)
        circ.x(0)
        return circ
