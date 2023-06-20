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

from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.exceptions import QiskitError

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from qiskit_experiments.curve_analysis import ParameterRepr, OscillationAnalysis
from qiskit_experiments.warnings import qubit_deprecate
from qiskit_experiments.library.characterization import CrossResRabiAnalysis


class Rabi(BaseExperiment, RestlessMixin):
    r"""An experiment that scans a pulse amplitude to calibrate rotations on the :math:`|0\rangle`
    <-> :math:`|1\rangle` transition.

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

    # section: manual
        :ref:`Rabi Calibration`

        See also `Qiskit Textbook <https://qiskit.org/textbook/ch-quantum-hardware/\
        calibrating-qubits-pulse.html>`_
        for the pulse level programming of a Rabi experiment.

    # section: analysis_ref
        :class:`~qiskit_experiments.curve_analysis.OscillationAnalysis`
    """

    __gate_name__ = "Rabi"
    __outcome__ = "rabi_rate"

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

    @qubit_deprecate()
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        amplitudes: Optional[Iterable[float]] = None,
        backend: Optional[Backend] = None,
    ):
        """Initialize a Rabi experiment on the given qubit.

        Args:
            physical_qubits: List with the qubit on which to run the Rabi experiment.
            schedule: The schedule that will be used in the Rabi experiment. This schedule
                should have one free parameter namely the amplitude.
            amplitudes: The pulse amplitudes that one wishes to scan. If this variable is not
                specified it will default to :code:`np.linspace(-0.95, 0.95, 51)`.
            backend: Optional, the backend to run the experiment on.
        """
        super().__init__(physical_qubits, analysis=OscillationAnalysis(), backend=backend)

        self.analysis.set_options(
            result_parameters=[ParameterRepr("freq", self.__outcome__)],
            normalization=True,

        )
        self.analysis.plotter.set_figure_options(
            xlabel="Amplitude",
            ylabel="Signal (arb. units)",
        )

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

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata


class EFRabi(Rabi):
    r"""An experiment that scans the amplitude of a pulse inducing rotations on the
     :math:`|1\rangle` <-> :math:`|2\rangle` transition.

    # section: overview

        This experiment is a subclass of the :class:`Rabi` experiment but takes place between
        the first and second excited state. An initial X gate populates the first excited state.
        The Rabi pulse is applied on the :math:`|1\rangle` <-> :math:`|2\rangle` transition
        (sometimes also labeled the e <-> f transition). The necessary frequency shift (typically
        the qubit anharmonicity) is given through the pulse schedule given at initialization. The
        schedule is then also stored in the experiment options. The circuits are of the form:

        .. parsed-literal::

                       ┌───┐┌───────────┐ ░ ┌─┐
                  q_0: ┤ X ├┤ Rabi(amp) ├─░─┤M├
                       └───┘└───────────┘ ░ └╥┘
            measure: 1/══════════════════════╩═
                                             0

    """

    __outcome__ = "rabi_rate_12"

    def _pre_circuit(self) -> QuantumCircuit:
        """A circuit with operations to perform before the Rabi."""
        circ = QuantumCircuit(1)
        circ.x(0)
        return circ


class CrossResRabi(BaseExperiment):
    """An experiment that scans a cross resonance amplitude to calibrate rotations angle.

    # section: overview

        This experiment takes a user provided schedule for cross resonance,
        which is attached to experiment circuits.

        .. parsed-literal::

                 ┌──────────────────────────┐
            q_0: ┤0                         ├───
                 │  CrossResRabi(amplitude) │┌─┐
            q_1: ┤1                         ├┤M├
                 └──────────────────────────┘└╥┘
            c: 1/═════════════════════════════╩═
                                              0

        This experiment measures only target qubit.

    # section: example

        User can build parameterized schedule and provide as follows:

        .. code-block:: python

            from qiskit_experiments.library import CrossResRabi
            from qiskit import pulse
            from qiskit.providers.fake_provider import FakeHanoiV2
            from math import pi

            amp_param = CrossResRabi.parameter
            backend = FakeHanoiV2()

            with pulse.build(default_alignment="sequential") as ecr:
                pulse.play(
                    pulse.GaussianSquare(
                        800, amp_param, 64, risefall_sigma_ratio=2, angle=0
                        ),
                    pulse.ControlChannel(0),
                )
                pulse.call(backend.target["x"][(0, )].calibration)
                pulse.play(
                    pulse.GaussianSquare(
                        800, amp_param, 64, risefall_sigma_ratio=2, angle=pi
                        ),
                    pulse.ControlChannel(0),
                )

            exp = CrossResRabi((0,), backend, schedule=ecr)

        Note that this example code attaches an echoed cross resonance sequence to the experiment.
        The sign of controlled rotation terms (ZX, ZY) is control qubit state dependent,
        and it may constructively or destructively interfere with local terms (IX, IY).
        Since the echo sequence eliminates the local terms, above example code may give you
        state independent Rabi rate.

    # section: analysis_ref
        :class:`~qiskit_experiments.curve_analysis.OscillationAnalysis`

    """

    parameter = Parameter("amplitude")
    """Qiskit Parameter object used to create parameterized circuits."""

    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Optional[Backend] = None,
        **experiment_options,
    ):
        """Create new experiment.

        Args:
            physical_qubits: Two element sequence of control and target qubit index.
            backend: Optional, the backend to run the experiment on.
            experiment_options: See ``self.default_experiment_options`` for details.
        """
        super().__init__(
            physical_qubits=physical_qubits,
            analysis=CrossResRabiAnalysis(),
            backend=backend,
        )
        if experiment_options:
            self.set_experiment_options(**experiment_options)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            amplitudes (list[float]): The list of amplitudes that will be scanned
                in the experiment. If not set, then ``num_amplitudes``
                evenly spaced amplitudes between ``min_amplitude`` and
                ``max_amplitude`` are used. If ``amplitudes`` is set,
                these experiment options are ignored.
            min_amplitude (float): Minimum amplitude to use.
            max_amplitude (float): Maximum amplitude to use.
            num_amplitudes (int): Number of circuits to use for parameter scan.
            schedule (ScheduleBlock): The Schedule that will be used in the
                experiment. This schedule must be parameterized with the
                parameter object ``CrossResRabi.parameter``.
        """
        options = super()._default_experiment_options()
        options.update_options(
            amplitudes=None,
            min_amplitude=-0.7,
            max_amplitude=0.7,
            num_amplitudes=51,
            schedule=None,
        )
        return options

    def parameters(self) -> np.ndarray:
        """Return parameters to scan."""
        opt = self.experiment_options

        if self.experiment_options.amplitudes:
            return np.asarray(opt.amplitudes, dtype=float)
        return np.linspace(opt.min_amplitude, opt.max_amplitude, opt.num_amplitudes)

    def parameterized_circuits(self) -> Tuple[QuantumCircuit, ...]:
        """Return parameterized circuits."""
        opt = self.experiment_options
        cr_rabi_gate = Gate("CrossResRabi", 2, [self.parameter])

        if opt.schedule is None:
            raise RuntimeError("Cross resonance schedule is not set in the experiment options.")

        if self.parameter not in opt.schedule.parameters:
            raise RuntimeError(
                f"Specified schedule {opt.schedule.name} might be parameterized with "
                "parameter objects that this experiment cannot recognize. "
                f"Please build your schedule with '{self.__class__.__name__}.parameter.'"
            )

        circuit = QuantumCircuit(2, 1)
        circuit.append(cr_rabi_gate, [0, 1])
        circuit.measure(1, 0)
        circuit.add_calibration(cr_rabi_gate, self.physical_qubits, opt.schedule)

        return (circuit,)

    def circuits(self) -> List[QuantumCircuit]:
        temp_circ = self.parameterized_circuits()[0]

        circs = []
        for param in self.parameters():
            assigned = temp_circ.assign_parameters({self.parameter: param}, inplace=False)
            assigned.metadata = {"xval": param}
            circs.append(assigned)

        return circs
