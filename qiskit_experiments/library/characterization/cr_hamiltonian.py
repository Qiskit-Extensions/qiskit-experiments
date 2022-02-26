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
Cross resonance Hamiltonian tomography.
"""

from typing import List, Tuple, Iterable, Optional, Type

import numpy as np
from qiskit import pulse, circuit, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
from qiskit.providers import Backend
from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis import CrossResonanceHamiltonianAnalysis


class CrossResonanceHamiltonian(BaseExperiment):
    r"""Cross resonance Hamiltonian tomography experiment.

    # section: overview

        This experiment assumes the two qubit Hamiltonian in the form

        .. math::

            H = \frac{I \otimes A}{2} + \frac{Z \otimes B}{2}

        where :math:`A` and :math:`B` are linear combinations of
        the Pauli operators :math:`\in {X, Y, Z}`.
        The coefficient of each Pauli term in the Hamiltonian
        can be estimated with this experiment.

        This experiment is performed by stretching the pulse duration of a cross resonance pulse
        and measuring the target qubit by projecting onto the x, y, and z bases.
        The control qubit state dependent (controlled-) Rabi oscillation on the
        target qubit is observed by repeating the experiment with the control qubit
        both in the ground and excited states. The fit for the oscillations in the
        three bases with the two control qubit preparations tomographically
        reconstructs the Hamiltonian in the form shown above.
        See Ref. [1] for more details.

        More specifically, the following circuits are executed in this experiment.

        .. parsed-literal::

            (X measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ P ├┤0                   ├────────
                 └───┘│  cr_tone(duration) │┌───┐┌─┐
            q_1: ─────┤1                   ├┤ H ├┤M├
                      └────────────────────┘└───┘└╥┘
            c: 1/═════════════════════════════════╩═
                                                  0

            (Y measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ P ├┤0                   ├───────────────
                 └───┘│  cr_tone(duration) │┌─────┐┌───┐┌─┐
            q_1: ─────┤1                   ├┤ Sdg ├┤ H ├┤M├
                      └────────────────────┘└─────┘└───┘└╥┘
            c: 1/════════════════════════════════════════╩═
                                                         0

            (Z measurement)

                 ┌───┐┌────────────────────┐
            q_0: ┤ P ├┤0                   ├───
                 └───┘│  cr_tone(duration) │┌─┐
            q_1: ─────┤1                   ├┤M├
                      └────────────────────┘└╥┘
            c: 1/════════════════════════════╩═
                                             0

        The ``P`` gate on the control qubit (``q_0``) indicates the state preparation.
        Since this experiment requires two sets of sub experiments with the control qubit in the
        excited and ground state, ``P`` will become ``X`` gate or just be omitted, respectively.
        Here ``cr_tone`` is implemented by a single cross resonance tone
        driving the control qubit at the frequency of the target qubit.
        The pulse envelope is the flat-topped Gaussian implemented by the parametric pulse
        :py:class:`~qiskit.pulse.library.parametric_pulses.GaussianSquare`.

        This experiment scans the flat-top width of the :py:class:`~qiskit.pulse.library.\
        parametric_pulses.GaussianSquare` envelope with the fixed rising and falling edges.
        The total pulse duration is implicitly computed to meet the timing constraints of
        the target backend. The edge duration is usually computed as

        .. math::

            \tau_{\rm edges} = 2 r \sigma,

        where the :math:`r` is the ratio of the actual edge duration to :math:`\sigma` of
        the Gaussian rising and falling edges. Note that actual edge duration is not
        identical to the net duration because of the smaller pulse amplitude of the edges.

        The net edge duration is an extra fitting parameter with initial guess

        .. math::

            \tau_{\rm edges}' = \sqrt{2 \pi} \sigma,

        which is derived by assuming a square edges with the full pulse amplitude.

    # section: analysis_ref
        :py:class:`CrossResonanceHamiltonianAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1603.04821

    # section: tutorial
        .. ref_website:: Qiskit Textbook 6.7,
            https://qiskit.org/textbook/ch-quantum-hardware/hamiltonian-tomography.html
    """

    # Number of CR pulses. The flat top duration per pulse is divided by this number.
    num_pulses = 1

    # pylint: disable=missing-class-docstring
    class CRPulseGate(circuit.Gate):
        def __init__(self, width: ParameterValueType):
            super().__init__("cr_gate", 2, [width])

    def __init__(
        self,
        qubits: Tuple[int, int],
        flat_top_widths: Iterable[float],
        backend: Optional[Backend] = None,
        cr_gate: Optional[Type[circuit.Gate]] = None,
        **kwargs,
    ):
        """Create a new experiment.

        Args:
            qubits: Two-value tuple of qubit indices on which to run tomography.
                The first index stands for the control qubit.
            flat_top_widths: The total duration of the square part of cross resonance pulse(s)
                to scan, in units of dt. The total pulse duration including Gaussian rising and
                falling edges is implicitly computed with experiment parameters ``sigma`` and
                ``risefall``.
            backend: Optional, the backend to run the experiment on.
            cr_gate: Optional, circuit gate instruction of cross resonance pulse.
            kwargs: Pulse parameters. See :meth:`experiment_options` for details.

        Raises:
            QiskitError: When ``qubits`` length is not 2.
        """
        super().__init__(qubits, analysis=CrossResonanceHamiltonianAnalysis(), backend=backend)

        if len(qubits) != 2:
            raise QiskitError(
                "Length of qubits is not 2. Please provide index for control and target qubit."
            )

        self.set_experiment_options(flat_top_widths=flat_top_widths, **kwargs)
        self._cr_gate = cr_gate or self.CRPulseGate

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            flat_top_widths (np.ndarray): The total duration of the square part of
                cross resonance pulse(s) to scan, in units of dt. This can start from zero and
                take positive real values representing the durations.
                Pulse edge effect is considered as an offset to the durations.
            amp (complex): Amplitude of the cross resonance tone.
            amp_t (complex): Amplitude of the cancellation or rotary drive on target qubit.
            sigma (float): Sigma of Gaussian rise and fall edges, in units of dt.
            risefall (float): Ratio of edge durations to sigma.
            cr_channel (PulseChannel): A channel index to apply cross resonance.
                This option is automatically filled when a backend is set.
            dt (float): A time resolution of control system.
                This option is automatically filled when a backend is set.
            granularity (int): A minimum chunk size of waveform memory.
                This option is automatically filled when a backend is set.
        """
        options = super()._default_experiment_options()
        options.flat_top_widths = None
        options.amp = 0.2
        options.amp_t = 0.0
        options.sigma = 64
        options.risefall = 2

        # These are automatically filled when backend is set
        options.cr_channel = 0
        options.dt = 1e-9
        options.granularity = 1

        return options

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)

        # Extract control channel index
        try:
            cr_channels = backend.configuration().control(self.physical_qubits)
            index = cr_channels[0].index
        except AttributeError:
            index = self.experiment_options.cr_channel

        # Extract time resolution
        try:
            dt_factor = backend.configuration().dt
        except AttributeError:
            dt_factor = self.experiment_options.dt

        # Extract pulse granularity
        try:
            granularity = backend.configuration().timing_constraints["granularity"]
        except (AttributeError, KeyError):
            granularity = self.experiment_options.granularity

        # Update experiment options
        self.set_experiment_options(dt=dt_factor, granularity=granularity, cr_channel=index)

    def _build_cr_circuit(self, pulse_gate: circuit.Gate) -> QuantumCircuit:
        """Single tone cross resonance.

        Args:
            pulse_gate: A pulse gate to represent a single cross resonance pulse.

        Returns:
            A circuit definition for the cross resonance pulse to measure.
        """
        cr_circuit = QuantumCircuit(2)
        cr_circuit.append(pulse_gate, [0, 1])

        return cr_circuit

    def _build_cr_schedule(self, flat_top_width: float) -> pulse.ScheduleBlock:
        """GaussianSquared cross resonance pulse.

        Args:
            flat_top_width: Total length of flat top part of the pulse in units of dt.

        Returns:
            A schedule definition for the cross resonance pulse to measure.
        """
        opt = self.experiment_options

        # Compute valid integer duration
        duration = flat_top_width + 2 * opt.sigma * opt.risefall
        valid_duration = int(opt.granularity * np.floor(duration / opt.granularity))

        with pulse.build(default_alignment="left", name="cr") as cross_resonance:

            # add cross resonance tone
            pulse.play(
                pulse.GaussianSquare(
                    duration=valid_duration,
                    amp=opt.amp,
                    sigma=opt.sigma,
                    width=flat_top_width,
                ),
                pulse.ControlChannel(opt.cr_channel),
            )
            # add cancellation tone
            if not np.isclose(opt.amp_t, 0.0):
                pulse.play(
                    pulse.GaussianSquare(
                        duration=valid_duration,
                        amp=opt.amp_t,
                        sigma=opt.sigma,
                        width=flat_top_width,
                    ),
                    pulse.DriveChannel(self.physical_qubits[1]),
                )
            else:
                pulse.delay(valid_duration, pulse.DriveChannel(self.physical_qubits[1]))

            # place holder for empty drive channels. this is necessary due to known pulse gate bug.
            pulse.delay(valid_duration, pulse.DriveChannel(self.physical_qubits[0]))

        return cross_resonance

    def set_experiment_options(self, **fields):
        super().set_experiment_options(**fields)

        # Effective length of Gaussian rising falling edges for fit guess (in dt).
        edge_duration = np.sqrt(2 * np.pi) * self.experiment_options.sigma * self.num_pulses

        init_guess = self.analysis.options.p0.copy()
        init_guess["t_off"] = edge_duration * self.experiment_options.dt

        self.analysis.set_options(p0=init_guess)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Returns:
            A list of :class:`QuantumCircuit`.

        Raises:
            AttributeError: When the backend doesn't report the time resolution of waveforms.
        """
        opt = self.experiment_options

        expr_circs = []
        for flat_top_width in opt.flat_top_widths:
            cr_schedule = self._build_cr_schedule(flat_top_width)
            cr_gate = self._cr_gate(flat_top_width)

            for control_state in (0, 1):
                for meas_basis in ("x", "y", "z"):
                    tomo_circ = QuantumCircuit(2, 1)

                    if control_state:
                        tomo_circ.x(0)

                    tomo_circ.compose(
                        other=self._build_cr_circuit(cr_gate),
                        qubits=[0, 1],
                        inplace=True,
                    )

                    if meas_basis == "x":
                        tomo_circ.h(1)
                    elif meas_basis == "y":
                        tomo_circ.sdg(1)
                        tomo_circ.h(1)
                    tomo_circ.measure(1, 0)

                    tomo_circ.metadata = {
                        "experiment_type": self.experiment_type,
                        "qubits": self.physical_qubits,
                        "xval": flat_top_width * opt.dt,  # in units of sec
                        "control_state": control_state,
                        "meas_basis": meas_basis,
                    }
                    if isinstance(cr_gate, self.CRPulseGate):
                        # Attach calibration if this is bare pulse gate
                        tomo_circ.add_calibration(
                            gate=cr_gate,
                            qubits=self.physical_qubits,
                            schedule=cr_schedule,
                        )

                    expr_circs.append(tomo_circ)

        return expr_circs


class EchoedCrossResonanceHamiltonian(CrossResonanceHamiltonian):
    r"""Echoed cross resonance Hamiltonian tomography experiment.

    # section: overview

        This is a variant of :py:class:`CrossResonanceHamiltonian`
        for which the experiment framework is identical but the
        cross resonance operation is realized as an echoed sequence
        to remove unwanted single qubit rotations. The cross resonance
        circuit looks like:

        .. parsed-literal::

                 ┌────────────────────┐  ┌───┐  ┌────────────────────┐
            q_0: ┤0                   ├──┤ X ├──┤0                   ├──────────
                 │  cr_tone(duration) │┌─┴───┴─┐│  cr_tone(duration) │┌────────┐
            q_1: ┤1                   ├┤ Rz(π) ├┤1                   ├┤ Rz(-π) ├
                 └────────────────────┘└───────┘└────────────────────┘└────────┘

        Here two ``cr_tone``s are applied where the latter one is with the
        control qubit state flipped and with a phase flip of the target qubit frame.
        This operation is equivalent to applying the ``cr_tone`` with a negative amplitude.
        The Hamiltonian for this decomposition has no IX and ZI interactions,
        and also a reduced IY interaction to some extent (not completely eliminated) [1].
        Note that the CR Hamiltonian tomography experiment cannot detect the ZI term.
        However, it is sensitive to the IX and IY terms.

    # section: reference
        .. ref_arxiv:: 1 2007.02925

    """
    num_pulses = 2

    def _build_cr_circuit(self, pulse_gate: circuit.Gate) -> QuantumCircuit:
        """Single tone cross resonance.

        Args:
            pulse_gate: A pulse gate to represent a single cross resonance pulse.

        Returns:
            A circuit definition for the cross resonance pulse to measure.
        """
        cr_circuit = QuantumCircuit(2)
        cr_circuit.append(pulse_gate, [0, 1])
        cr_circuit.x(0)
        cr_circuit.rz(np.pi, 1)
        cr_circuit.append(pulse_gate, [0, 1])
        cr_circuit.rz(-np.pi, 1)

        return cr_circuit
