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

"""Rough drag experiment."""

from typing import Iterable, List, Optional, Sequence
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from qiskit_experiments.library.characterization.analysis import DragCalAnalysis


class RoughDrag(RestlessMixin, BaseExperiment):
    r"""An experiment that scans the DRAG parameter to find the optimal value.

    # section: overview

        A Derivative Removal by Adiabatic Gate (DRAG) pulse is designed to minimize phase
        errors and leakage resulting from the presence of a neighbouring transition. DRAG
        is a standard pulse with an additional derivative component. The optimal value of
        the DRAG parameter, :math:`\beta`, is chosen to primarily minimize phase errors
        resulting from the AC Stark shift and potentially leakage errors. The DRAG pulse is

        .. math::

            f(t) = \Omega(t) + 1j \beta d/dt \Omega(t)

        Here, :math:`\Omega` is the envelop of the in-phase component of the pulse and
        :math:`\beta` is the strength of the quadrature which we refer to as the DRAG
        parameter and seek to calibrate in this experiment. The DRAG calibration will run
        several series of circuits. In a given circuit a Rp(β) - Rm(β) block is repeated
        :math:`N` times. Here, Rp is a rotation with a positive angle and Rm is the same rotation
        with a native angle and is implemented by the gate sequence Rz(π) - Rp(β) - Rz(π) where
        the Z rotations are virtual. As example the circuit of a single repetition, i.e.
        :math:`N=1`, is shown below.

        .. parsed-literal::

                       ┌───────┐┌───────┐┌───────┐┌───────┐ ░ ┌─┐
                  q_0: ┤ Rp(β) ├┤ Rz(π) ├┤ Rp(β) ├┤ Rz(π) ├─░─┤M├
                       └───────┘└───────┘└───────┘└───────┘ ░ └╥┘
            measure: 1/════════════════════════════════════════╩═
                                                               0

        The parameter β is scanned to find the value that minimizes the unwanted Z-rotation.
        Note that the analysis class requires this experiment to run with three repetition numbers.

    # section: analysis_ref
        :class:`DragCalAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=False, seed=101)

        .. jupyter-execute::

            import numpy as np
            from qiskit import pulse
            from qiskit.circuit import Parameter
            from qiskit_experiments.library import RoughDrag

            with pulse.build() as build_sched:
                pulse.play(pulse.Drag(160, 0.50, 40, Parameter("beta")), pulse.DriveChannel(0))

            exp = RoughDrag(physical_qubits=(0,),
                            schedule=build_sched,
                            betas = np.linspace(-4, 4, 51),
                            backend=backend,)
            exp.set_experiment_options(reps=[3, 5, 7])

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)

    # section: reference
        .. ref_arxiv:: 1 1011.1949
        .. ref_arxiv:: 2 0901.0534
        .. ref_arxiv:: 3 1509.05470

    # section: manual
        :ref:`DRAG Calibration`

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the rough drag experiment.

        Experiment Options:
            schedule (ScheduleBlock): The schedule of the rotation.
            reps (List[int]): The number of times the Rp - Rm gate sequence is repeated in
                each series. Note that this list must always have a length of three as
                otherwise the analysis class will not run.
            betas (Iterable): the values of the DRAG parameter to scan.
        """
        options = super()._default_experiment_options()
        options.schedule = None
        options.reps = [1, 3, 5]
        options.betas = np.linspace(-5, 5, 51)

        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        betas: Optional[Iterable[float]] = None,
        backend: Optional[Backend] = None,
    ):
        """Initialize a Drag experiment in the given qubit.

        Args:
            physical_qubits: Sequence containing the qubit for which to run the
                Drag calibration.
            schedule: The schedule to run. This schedule should have one free parameter
                corresponding to a DRAG parameter.
            betas: The values of the DRAG parameter to scan. If None is given the default range
                :code:`linspace(-5, 5, 51)` is used.
            backend: Optional, the backend to run the experiment on.

        Raises:
            QiskitError: If the schedule does not have a free parameter.
        """

        # Create analysis in finalize to reflect user change to reps
        super().__init__(physical_qubits, analysis=DragCalAnalysis(), backend=backend)

        if betas is not None:
            self.set_experiment_options(betas=betas)

        if len(schedule.parameters) != 1:
            raise QiskitError(
                f"Schedule {schedule} for {self.__class__.__name__} experiment must have "
                f"exactly one free parameter, found {schedule.parameters} parameters."
            )

        self.set_experiment_options(schedule=schedule)

    def _pre_circuit(self) -> QuantumCircuit:
        """A circuit with operations to perform before the Drag."""
        return QuantumCircuit(1)

    def circuits(self) -> List[QuantumCircuit]:
        """Create the circuits for the Drag calibration.

        Returns:
            circuits: The circuits that will run the Drag calibration.

        Raises:
            QiskitError: If the number of different repetition series is not three.
        """
        schedule = self.experiment_options.schedule

        beta = next(iter(schedule.parameters))

        # Note: If the pulse has a reserved name, e.g. x, which does not have parameters
        # then we cannot directly call the gate x and attach a schedule to it. Doing so
        # would results in QObj errors.
        drag_gate = Gate(name="Drag(" + schedule.name + ")", num_qubits=1, params=[beta])

        circuits = []
        for rep in self.experiment_options.reps:
            circuit = self._pre_circuit()
            for _ in range(rep):
                circuit.append(drag_gate, (0,))
                circuit.rz(np.pi, 0)
                circuit.append(drag_gate, (0,))
                circuit.rz(np.pi, 0)

            circuit.measure_active()

            circuit.add_calibration(
                "Drag(" + schedule.name + ")", self.physical_qubits, schedule, params=[beta]
            )

            for beta_val in self.experiment_options.betas:
                beta_val = float(np.round(beta_val, decimals=6))

                assigned_circuit = circuit.assign_parameters({beta: beta_val}, inplace=False)

                assigned_circuit.metadata = {
                    "xval": beta_val,
                    "nrep": rep,
                }

                circuits.append(assigned_circuit)

        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
