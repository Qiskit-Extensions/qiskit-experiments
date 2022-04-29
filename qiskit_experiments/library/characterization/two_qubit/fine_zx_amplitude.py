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

"""Fine two-qubits amplitude characterization experiment."""

from typing import Optional, Sequence
import numpy as np

from qiskit.circuit import Gate
from qiskit.providers.backend import Backend

from qiskit_experiments.data_processing import DataProcessor, nodes
from qiskit_experiments.framework import Options
from qiskit_experiments.library.characterization.fine_amplitude import FineAmplitude


class FineZXAmplitude(FineAmplitude):
    r"""A fine amplitude experiment for the :code:`RZXGate(np.pi / 2)`.

    # section: overview

        :class:`FineZXAmplitude` is a subclass of :class:`FineAmplitude` and is used to set
        the appropriate values for the default options to calibrate a :code:`RZXGate(np.pi / 2)`.

    # section: example

        To run this experiment the user will have to provide the instruction schedule
        map in the transpile options that contains the schedule for the experiment.

        ..code-block:: python

            qubits = (1, 2)
            inst_map = InstructionScheduleMap()
            inst_map.add("szx", qubits, my_schedule)

            fine_amp = FineZXAmplitude(qubits, backend)
            fine_amp.set_transpile_options(inst_map=inst_map)

        Here, :code:`my_schedule` is the pulse schedule that will implement the
        :code:`RZXGate(np.pi / 2)` rotation.
    """

    def __init__(self, qubits: Sequence[int], backend: Optional[Backend] = None):
        """Initialize the experiment."""

        # We cannot use RZXGate since it has a parameter so we redefine the gate.
        # Failing to do so causes issues with QuantumCircuit.calibrations.
        gate = Gate("szx", 2, [])

        super().__init__(qubits, gate, backend=backend, measurement_qubits=[qubits[1]])
        # Set default analysis options
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi / 2,
                "phase_offset": np.pi,
            },
            outcome="1",
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            add_cal_circuits (bool): If set to True then two circuits to calibrate 0 and 1 points
                will be added. This option is set to False by default for ``FineZXAmplitude``
                since the amplitude calibration can be achieved with two RZX gates and this is
                included in the repetitions.
            repetitions (List[int]): A list of the number of times that the gate is repeated.
        """
        options = super()._default_experiment_options()
        options.add_cal_circuits = False
        options.repetitions = [0, 1, 2, 3, 4, 5, 7, 9, 11, 13]
        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpile options for the fine amplitude experiment.

        Experiment Options:
            basis_gates: Set to :code:`["szx"]`.
            inst_map: The instruction schedule map that will contain the schedule for the
                Rzx(pi/2) gate. This schedule should be stored under the instruction name
                ``szx``.
        """
        options = super()._default_transpile_options()
        options.basis_gates = ["szx"]
        options.inst_map = None
        return options

    def enable_restless(
        self, rep_delay: Optional[float] = None, override_processor_by_restless: bool = True
    ):
        """Enable restless measurements.

        We wrap the method of the :class:`RestlessMixin` to readout both qubits. This forces
        the control qubit to be in either the 0 or 1 state before the next circuit starts
        since restless measurements do not reset qubits.
        """
        self.analysis.set_options(outcome="11")
        super().enable_restless(rep_delay, override_processor_by_restless)
        self._measurement_qubits = range(self.num_qubits)

    def _get_restless_processor(self) -> DataProcessor:
        """Marginalize the counts after the restless shot reordering."""
        return DataProcessor(
            "memory",
            [
                nodes.RestlessToCounts(self._num_qubits),
                nodes.MarginalizeCounts({1}),  # keep only the target.
                nodes.Probability("1"),
            ],
        )
