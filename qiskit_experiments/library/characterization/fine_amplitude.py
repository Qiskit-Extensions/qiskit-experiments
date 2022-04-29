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
from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from qiskit_experiments.library.characterization.analysis import FineAmplitudeAnalysis


class FineAmplitude(BaseExperiment, RestlessMixin):
    r"""Error amplifying fine amplitude calibration experiment.

    # section: overview

        The :class:`FineAmplitude` calibration experiment repeats N times a gate with a pulse
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
        pi-pulses. Importantly, the resulting data is analyzed by a fit to a cosine function in
        which we try to determine the over/under rotation given an intended rotation angle per
        gate which must also be specified by the user.

        Error amplifying experiments are most sensitive to angle errors when we measure points along
        the equator of the Bloch sphere. This is why users should insert a square-root of X pulse
        before running calibrations for :math:`\pm\pi` rotations. When all data points are close to
        the equator, it is difficult for a fitter to infer the overall scale of the error. When
        calibrating a :math:`pi` rotation, one can use ``add_xp_circuit = True`` to insert one
        circuit that puts the qubit in the excited state to set the scale for the other circuits.
        Furthermore, when running calibrations for :math:`\pm\pi/2` rotations users are advised
        to use an odd number of repetitions, e.g. [1, 2, 3, 5, 7, ...] to ensure that the ideal
        points are on the equator of the Bloch sphere. Note the presence of two repetitions which
        allows us to prepare the excited state. Therefore, ``add_xp_circuit = True`` is not needed
        in this case.

    # section: example

        The steps to run a fine amplitude experiment are

        .. code-block:: python

            qubit = 3
            amp_cal = FineAmplitude(qubit, SXGate())
            amp_cal.set_experiment_options(
                angle_per_gate=np.pi/2,
                add_xp_circuit=False,
                add_sx=False
            )
            amp_cal.run(backend)

        Note that there are subclasses of :class:`FineAmplitude` such as :class:`FineSXAmplitude`
        that set the appropriate options by default.

    # section: analysis_ref
        :py:class:`FineAmplitudeAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1504.06597

    # section: tutorial
        :doc:`/tutorials/fine_calibrations`

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that the gate is repeated.
            gate_type (Gate): This is a gate class such as XGate, so that one can obtain a gate
                by doing :code:`options.gate_class()`.
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
        qubits: Sequence[int],
        gate: Gate,
        backend: Optional[Backend] = None,
        measurement_qubits: Sequence[int] = None,
    ):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            qubits: The qubit(s) on which to run the fine amplitude calibration experiment.
            gate: The gate that will be repeated.
            backend: Optional, the backend to run the experiment on.
            measurement_qubits: The qubits in the given physical qubits that need to
                be measured.
        """
        super().__init__(qubits, analysis=FineAmplitudeAnalysis(), backend=backend)
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
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "xval": add_x,
                "unit": "gate number",
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
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "xval": repetition,
                "unit": "gate number",
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
    """

    def __init__(self, qubit: int, backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__([qubit], XGate(), backend=backend)
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
    """

    def __init__(self, qubit: int, backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__([qubit], SXGate(), backend=backend)
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
