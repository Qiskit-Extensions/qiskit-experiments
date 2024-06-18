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


class Rabi(RestlessMixin, BaseExperiment):
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

        See also the `Qiskit Textbook
        <https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware-pulses/calibrating-qubits-pulse.ipynb>`_
        for the pulse level programming of a Rabi experiment.

    # section: analysis_ref
        :class:`~qiskit_experiments.curve_analysis.OscillationAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-0.25e9, 1e9, 0.8e9, 1e4, noise=True, seed=199)

        .. jupyter-execute::

            import numpy as np
            from qiskit import pulse
            from qiskit.circuit import Parameter
            from qiskit_experiments.library import Rabi

            with pulse.build() as build_sched:
                pulse.play(pulse.Gaussian(160, Parameter("amp"), 40), pulse.DriveChannel(0))

            exp = Rabi(physical_qubits=(0,),
                       schedule=build_sched,
                       amplitudes=np.linspace(-0.1, 0.1, 21),
                       backend=backend,)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)
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
            # casting is needed because for amplitude '0', np.round method return datatype of int32
            # which isn't serializable in the metadata.
            amp = float(np.round(amp, decimals=6))
            assigned_circ = circuit.assign_parameters({param: amp}, inplace=False)
            assigned_circ.metadata = {"xval": amp}

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
    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9, -0.25e9, 1e9, 0.8e9, 1e4, noise=True, seed=198)

        .. jupyter-execute::

            import numpy as np
            from qiskit import pulse
            from qiskit.circuit import Parameter
            from qiskit_experiments.library import EFRabi

            with pulse.build() as build_sched:
                pulse.play(pulse.Gaussian(160, Parameter("amp"), sigma=40), pulse.DriveChannel(0))

            exp = EFRabi(physical_qubits=(0,),
                         backend=backend,
                         schedule=build_sched,
                         amplitudes=np.linspace(-0.1, 0.1, 21),)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)
    """

    __outcome__ = "rabi_rate_12"

    def _pre_circuit(self) -> QuantumCircuit:
        """A circuit with operations to perform before the Rabi."""
        circ = QuantumCircuit(1)
        circ.x(0)
        return circ
