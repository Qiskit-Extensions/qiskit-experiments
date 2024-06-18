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

"""Fine amplitude characterization experiment."""

from typing import List, Optional, Sequence
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, SXGate
from qiskit.providers.backend import Backend
from qiskit_experiments.data_processing import DataProcessor, nodes
from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from qiskit_experiments.library.characterization.analysis import FineAmplitudeAnalysis


class FineAmplitude(RestlessMixin, BaseExperiment):
    r"""An experiment to determine the optimal pulse amplitude by amplifying gate errors.

    # section: overview

        The :class:`FineAmplitude` experiment repeats N times a gate with a pulse
        to amplify the under-/over-rotations in the gate to determine the optimal amplitude.
        The circuits are therefore of the form:

        .. parsed-literal::

                       ┌──────┐       ┌──────┐ ░ ┌─┐
                  q_0: ┤ Gate ├─ ... ─┤ Gate ├─░─┤M├
                       └──────┘       └──────┘ ░ └╥┘
            measure: 1/═════════ ... ═════════════╩═
                                                  0

        Here, Gate is the name of the gate which will be repeated. The user can optionally add a
        square-root of X pulse before the gates are repeated. This square-root of X pulse allows
        the analysis to differentiate between over rotations and under rotations in the case of
        :math:`\pi`-pulses. Importantly, the resulting data is analyzed by a fit to a cosine function in
        which we try to determine the over/under rotation given an intended rotation angle per
        gate which must also be specified by the user.

        Error amplifying experiments are most sensitive to angle errors when we measure points along
        the equator of the Bloch sphere. This is why users should insert a square-root of X pulse
        before running calibrations for :math:`\pm\pi` rotations. When all data points are close to
        the equator, it is difficult for a fitter to infer the overall scale of the error. When
        calibrating a :math:`\pi` rotation, one can use ``add_xp_circuit = True`` to insert one
        circuit that puts the qubit in the excited state to set the scale for the other circuits.
        Furthermore, when running calibrations for :math:`\pm\pi/2` rotations users are advised
        to use an odd number of repetitions, e.g. [1, 2, 3, 5, 7, ...] to ensure that the ideal
        points are on the equator of the Bloch sphere. Note the presence of two repetitions which
        allows us to prepare the excited state. Therefore, ``add_xp_circuit = True`` is not needed
        in this case.

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e6, noise=True, seed=185)

        .. jupyter-execute::

            import numpy as np
            from qiskit.circuit.library import XGate
            from qiskit_experiments.library import FineAmplitude

            exp = FineAmplitude(physical_qubits=(0,), gate=XGate(), backend=backend)
            exp.analysis.set_options(fixed_parameters={"angle_per_gate" : np.pi, "phase_offset" : np.pi})

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)

    # section: analysis_ref
        :class:`FineAmplitudeAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1504.06597

    # section: manual
        :ref:`fine-amplitude-cal`

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that the gate is repeated.
            gate (Gate): This is a gate class such as XGate, so that one can obtain a gate
                by doing :code:`options.gate()`.
            normalization (bool): If set to True the DataProcessor will normalized the
                measured signal to the interval [0, 1]. Defaults to True.
            add_cal_circuits (bool): If set to True then two circuits to calibrate 0 and 1 points
                will be added. These circuits are often needed to properly calibrate the amplitude
                of the ping-pong oscillation that encodes the errors. This helps account for
                state preparation and measurement errors.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(1, 15))
        options.gate = None
        options.normalization = True
        options.add_cal_circuits = True

        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        gate: Gate,
        backend: Optional[Backend] = None,
        measurement_qubits: Sequence[int] = None,
    ):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            physical_qubits: Sequence containing qubit(s) on which to run the
                fine amplitude calibration experiment.
            gate: The gate that will be repeated.
            backend: Optional, the backend to run the experiment on.
            measurement_qubits: The qubits in the given physical qubits that need to
                be measured.
        """
        super().__init__(physical_qubits, analysis=FineAmplitudeAnalysis(), backend=backend)
        self.set_experiment_options(gate=gate)

        if measurement_qubits is not None:
            self._measurement_qubits = [self.physical_qubits.index(q) for q in measurement_qubits]
        else:
            self._measurement_qubits = range(self.num_qubits)

    def _spam_cal_circuits(self, meas_circuit: QuantumCircuit) -> List[QuantumCircuit]:
        """This method returns the calibration circuits.

        Calibration circuits allow the experiment to overcome state preparation and
        measurement errors which cause ideal probabilities to be below 1.

        Args:
            meas_circuit: The measurement circuit, so that we only apply x gates to the
                measured qubits.

        Returns:
            Two circuits that calibrate the spam errors for the 0 and 1 state.
        """
        cal_circuits = []

        for add_x in [0, 1]:
            circ = QuantumCircuit(self.num_qubits, meas_circuit.num_clbits)

            if add_x:
                qubits = meas_circuit.get_instructions("measure")[0][1]
                circ.x(qubits)

            circ.compose(meas_circuit, inplace=True)

            circ.metadata = {
                "xval": add_x,
                "series": "spam-cal",
            }

            cal_circuits.append(circ)

        return cal_circuits

    def _pre_circuit(self, num_clbits: int) -> QuantumCircuit:
        """Return a preparation circuit.

        This method can be overridden by subclasses e.g. to calibrate gates on
        transitions other than the 0 <-> 1 transition.
        """
        return QuantumCircuit(self.num_qubits, num_clbits)

    def _measure_circuit(self) -> QuantumCircuit:
        """Create the measurement part of the quantum circuit.

        Sub-classes may override this function.

        Returns:
            A quantum circuit which defines the qubits that will be measured.
        """
        circuit = QuantumCircuit(self.num_qubits, len(self._measurement_qubits))

        for idx, qubit in enumerate(self._measurement_qubits):
            circuit.measure(qubit, idx)

        return circuit

    def circuits(self) -> List[QuantumCircuit]:
        """Create the circuits for the fine amplitude calibration experiment.

        Returns:
            A list of circuits with a variable number of gates.

        Raises:
            CalibrationError: If the analysis options do not contain the angle_per_gate.
        """
        repetitions = self.experiment_options.get("repetitions")

        qubits = range(self.num_qubits)
        meas_circ = self._measure_circuit()
        pre_circ = self._pre_circuit(meas_circ.num_clbits)

        if self.experiment_options.add_cal_circuits:
            circuits = self._spam_cal_circuits(meas_circ)
        else:
            circuits = []

        for repetition in repetitions:
            circuit = QuantumCircuit(self.num_qubits, meas_circ.num_clbits)

            # Add pre-circuit
            circuit.compose(pre_circ, qubits, range(meas_circ.num_clbits), inplace=True)

            for _ in range(repetition):
                circuit.append(self.experiment_options.gate, qubits)

            # Add the measurement part of the circuit
            circuit.compose(meas_circ, qubits, range(meas_circ.num_clbits), inplace=True)

            circuit.metadata = {
                "xval": repetition,
                "series": 1,
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


class FineXAmplitude(FineAmplitude):
    r"""A fine amplitude experiment with all the options set for the :math:`\pi`-rotation.

    # section: overview

        :class:`FineXAmplitude` is a subclass of :class:`FineAmplitude` and is used to set
        the appropriate values for the default options.

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=True, seed=198)

        .. jupyter-execute::

            from qiskit_experiments.library import FineXAmplitude

            exp = FineXAmplitude(physical_qubits=(0,), backend=backend)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)
    """

    def __init__(self, physical_qubits: Sequence[int], backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__(physical_qubits, XGate(), backend=backend)
        # Set default analysis options
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi,
                "phase_offset": np.pi / 2,
            }
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            gate (Gate): Gate to characterize. Defaults to an XGate.
        """
        options = super()._default_experiment_options()
        options.gate = XGate()
        return options

    def _pre_circuit(self, num_clbits: int) -> QuantumCircuit:
        """The preparation circuit is an sx gate to move to the equator of the Bloch sphere."""
        circuit = QuantumCircuit(self.num_qubits, num_clbits)
        circuit.sx(0)
        return circuit


class FineSXAmplitude(FineAmplitude):
    r"""A fine amplitude experiment with all the options set for the :math:`\pi/2`-rotation.

    # section: overview

        :class:`FineSXAmplitude` is a subclass of :class:`FineAmplitude` and is used to set
        the appropriate values for the default options.

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=True, seed=198)

        .. jupyter-execute::

            from qiskit_experiments.library import FineSXAmplitude

            exp = FineSXAmplitude(physical_qubits=(0,), backend=backend)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)
    """

    def __init__(self, physical_qubits: Sequence[int], backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__(physical_qubits, SXGate(), backend=backend)
        # Set default analysis options
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi / 2,
                "phase_offset": np.pi,
            }
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            gate (Gate): FineSXAmplitude calibrates an SXGate.
            add_cal_circuits (bool): If set to True then two circuits to calibrate 0 and 1 points
                will be added. This option is set to False by default for ``FineSXAmplitude``
                since the amplitude calibration can be achieved with two SX gates and this is
                included in the repetitions.
            repetitions (List[int]): By default the repetitions take on odd numbers for
                :math:`\pi/2` target angles as this ideally prepares states on the equator of
                the Bloch sphere. Note that the repetitions include two repetitions which
                plays the same role as including a circuit with an X gate.
        """
        options = super()._default_experiment_options()
        options.gate = SXGate()
        options.add_cal_circuits = False
        options.repetitions = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]

        return options


class FineZXAmplitude(FineAmplitude):
    r"""A fine amplitude experiment for the :code:`RZXGate(np.pi / 2)`.

    # section: overview

        :class:`FineZXAmplitude` is a subclass of :class:`FineAmplitude` and is used to set
        the appropriate values for the default options to calibrate a :code:`RZXGate(np.pi / 2)`.

    # section: example

        To run this experiment, the user will have to provide the instruction schedule
        map in the transpile options that contains the schedule for the experiment.

        .. code-block:: python

            qubits = (1, 2)
            inst_map = InstructionScheduleMap()
            inst_map.add("szx", qubits, my_schedule)

            fine_amp = FineZXAmplitude(qubits, backend)
            fine_amp.set_transpile_options(inst_map=inst_map)

        Here, :code:`my_schedule` is the pulse schedule that will implement the
        :code:`RZXGate(np.pi / 2)` rotation.
    """

    def __init__(self, physical_qubits: Sequence[int], backend: Optional[Backend] = None):
        """Initialize the experiment."""

        # We cannot use RZXGate since it has a parameter so we redefine the gate.
        # Failing to do so causes issues with QuantumCircuit.calibrations.
        gate = Gate("szx", 2, [])

        super().__init__(
            physical_qubits, gate, backend=backend, measurement_qubits=[physical_qubits[1]]
        )
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
        self,
        rep_delay: Optional[float] = None,
        override_processor_by_restless: bool = True,
        suppress_t1_error: bool = False,
    ):
        """Enable restless measurements.

        We wrap the method of the :class:`.RestlessMixin` to readout both qubits. This forces
        the control qubit to be in either the 0 or 1 state before the next circuit starts
        since restless measurements do not reset qubits.

        Args:
            rep_delay: The repetition delay. This is the delay between a measurement
                and the subsequent quantum circuit. Since the backends have
                dynamic repetition rates, the repetition delay can be set to a small
                value which is required for restless experiments. Typical values are
                1 us or less.
            override_processor_by_restless: If False, a data processor that is specified in the
                analysis options of the experiment is not overridden by the restless data
                processor. The default is True.
            suppress_t1_error: If True, the default is False, then no error will be raised when
                ``rep_delay`` is larger than the T1 times of the qubits. Instead, a warning will
                be logged as restless measurements may have a large amount of noise.
        """
        self.analysis.set_options(outcome="11")
        super().enable_restless(rep_delay, override_processor_by_restless, suppress_t1_error)
        self._measurement_qubits = range(self.num_qubits)

    def _get_restless_processor(self, meas_level: int = 2) -> DataProcessor:
        """Marginalize the counts after the restless shot reordering."""
        return DataProcessor(
            "memory",
            [
                nodes.RestlessToCounts(self._num_qubits),
                nodes.MarginalizeCounts({1}),  # keep only the target.
                nodes.Probability("1"),
            ],
        )
