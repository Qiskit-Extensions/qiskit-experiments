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

"""Fine DRAG characterization experiment."""

from typing import List, Optional, Sequence
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, SXGate
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from qiskit_experiments.curve_analysis.standard_analysis import ErrorAmplificationAnalysis


class FineDrag(RestlessMixin, BaseExperiment):
    r"""An experiment that performs fine characterizations of DRAG pulse coefficients.

    # section: overview

        :class:`FineDrag` runs fine DRAG characterization experiments (see :class:`.RoughDrag`
        for the definition of DRAG pulses). Fine DRAG proceeds by iterating the gate sequence
        Rp - Rm where Rp is a rotation around an axis and Rm is the same rotation but in the
        opposite direction and is implemented by the gates Rz - Rp - Rz where the Rz gates
        are virtual Z-rotations, see Ref. [1]. The executed circuits are of the form

        .. parsed-literal::

                    ┌─────┐┌────┐┌───────┐┌────┐┌───────┐     ┌──────┐ ░ ┌─┐
               q_0: ┤ Pre ├┤ Rp ├┤ Rz(π) ├┤ Rp ├┤ Rz(π) ├ ... ┤ Post ├─░─┤M├
                    └─────┘└────┘└───────┘└────┘└───────┘     └──────┘ ░ └╥┘
            meas: 1/══════════════════════════════════════════════════════╩═
                                                                          0

        Here, "Pre" and "Post" designate gates that may be pre-appended and and post-appended,
        respectively, to the repeated sequence of Rp - Rz - Rp - Rz gates. When calibrating
        a pulse with a target rotation angle of π the Pre and Post gates are Id and RYGate(π/2),
        respectively. When calibrating a pulse with a target rotation angle of π/2 the Pre and
        Post gates are RXGate(π/2) and RYGate(π/2), respectively.

        We now describe what this experiment corrects by following Ref. [2]. We follow equations
        4.30 and onwards of Ref. [2] which state that the first-order corrections to the control
        fields are

        .. math::

            \begin{align}
                \bar{\Omega}_x^{(1)}(t) = &\, 2\dot{s}^{(1)}_{x,0,1}(t) \\
                \bar{\Omega}_y^{(1)}(t) = &\, 2\dot{s}^{(1)}_{y,0,1}(t)
                - s_{z,1}^{(1)}(t)t_g\Omega_x(t) \\
                \bar{\delta}^{(1)}(t) = &\, \dot{s}_{z,1}^{(1)}(t) + 2s^{(1)}_{y,0,1}(t)t_g\Omega_x(t)
                 + \frac{\lambda_1^2 t_g^2 \Omega_x^2(t)}{4}
            \end{align}


        Here, the :math:`s` terms are coefficients of the expansion of an operator :math:`S(t)`
        that generates a transformation that keeps the qubit sub-space isolated from the
        higher-order states. :math:`t_g` is the gate time, :math:`\Omega_x(t)` is the pulse envelope
        on the in-phase component of the drive and :math:`\lambda_1` is a parameter of the Hamiltonian.
        For additional details please see Ref. [2].
        As in Ref. [2] we now set :math:`s^{(1)}_{x,0,1}` and :math:`s^{(1)}_{z,1}` to zero
        and set :math:`s^{(1)}_{y,0,1}` to :math:`-\lambda_1^2 t_g\Omega_x(t)/8`. This
        results in a Z angle rotation rate of :math:`\bar{\delta}^{(1)}(t)=0` in the equations
        above and defines the value for the ideal :math:`\beta` parameter. In Qiskit pulse, the
        definition of the DRAG pulse is

        .. math::

            \Omega(t) = \Omega_x(t) + i\beta\,\dot{\Omega}_x(t)\quad\Longrightarrow\quad
            \Omega_y(t)= \beta\,\dot{\Omega}_x(t)

        which implies that :math:`-\lambda_1^2 t_g/4` is the ideal :math:`\beta` value.
        We now assume that there is a small error :math:`{\rm d}\beta` in :math:`\beta` such
        that the instantaneous Z-angle error induced by a single pulse is

        .. math::

            \bar\delta(t) = {\rm d}\beta\, \Omega^2_x(t)


        We can integrate :math:`\bar{\delta}(t)`, i.e. the instantaneous :math:`Z`-angle rotation error,
        to obtain the total rotation angle error per pulse, :math:`{\rm d}\theta`:

        .. math::

           {\rm d}\theta = \int\bar\delta(t){\rm d}t = {\rm d}\beta \int\Omega^2_x(t){\rm d}t

        If we assume a Gaussian pulse, i.e. :math:`\Omega_x(t)=A\exp[-t^2/(2\sigma^2)]`
        then the integral of :math:`\Omega_x^2(t)` in the equation above results in
        :math:`A^2\sigma\sqrt{\pi}`. Furthermore, the integral of :math:`\Omega_x(t)` is
        :math:`A\sigma\sqrt{\pi/2}=\theta_\text{target}`, where :math:`\theta_\text{target}`
        is the target rotation angle, i.e. the area under the pulse. This last point allows
        us to rewrite :math:`A^2\sigma\sqrt{\pi}` as
        :math:`\theta^2_\text{target}/(2\sigma\sqrt{\pi})`. The total :math:`Z` angle error per pulse
        is therefore

        .. math::

           {\rm d}\theta=
            \int\bar\delta(t){\rm d}t={\rm d}\beta\,\frac{\theta^2_\text{target}}{2\sigma\sqrt{\pi}}

        Here, :math:`{\rm d}\theta` is the :math:`Z` angle error per pulse. The qubit population
        produced by the gate sequence shown above is used to measure :math:`{\rm d}\theta`. Indeed,
        each gate pair Rp - Rm will produce a small unwanted :math:`Z`-rotation out of the
        :math:`ZX` plane with a magnitude :math:`2\,{\rm d}\theta`. The total rotation out of the
        :math:`ZX` plane is then mapped to a qubit population by the final Post gate. Inverting the
        relation above after cancelling out the factor of two due to the Rp - Rm pulse pair yields
        the error in :math:`\beta` that produced the rotation error :math:`{\rm d}\theta` as

        .. math::

            {\rm d}\beta=\frac{\sqrt{\pi}\,{\rm d}\theta\sigma}{ \theta_\text{target}^2}.

        This is the correction formula in the FineDRAG Updater.

    # section: analysis_ref
        :class:`.ErrorAmplificationAnalysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=False, seed=199)

        .. jupyter-execute::

            from qiskit.circuit.library import XGate
            from qiskit_experiments.library.characterization import FineDrag

            exp = FineDrag(physical_qubits=(0,), gate=XGate(), backend=backend)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)

    # section: reference
        .. ref_arxiv:: 1 1612.00858
        .. ref_arxiv:: 2 1011.1949
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that Rp - Rm gate sequence
                is repeated.
            schedule (ScheduleBlock): The schedule for the plus rotation.
            gate (Gate): This is the gate such as XGate() that will be in the circuits.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(20))
        options.schedule = None
        options.gate = None

        return options

    def __init__(
        self, physical_qubits: Sequence[int], gate: Gate, backend: Optional[Backend] = None
    ):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            physical_qubits: List containing the qubit on which to run the fine
                amplitude calibration experiment.
            gate: The gate that will be repeated.
            backend: Optional, the backend to run the experiment on.
        """
        analysis = ErrorAmplificationAnalysis()
        analysis.set_options(
            normalization=True,
            fixed_parameters={
                "angle_per_gate": 0.0,
                "phase_offset": np.pi / 2,
                "amp": 1.0,
            },
        )

        super().__init__(physical_qubits, analysis=analysis, backend=backend)
        self.set_experiment_options(gate=gate)

    @staticmethod
    def _pre_circuit() -> QuantumCircuit:
        """Return the quantum circuit to apply before repeating the Rp - Rz - Rp - Rz gates."""
        return QuantumCircuit(1)

    @staticmethod
    def _post_circuit() -> QuantumCircuit:
        """Return the quantum circuit to apply after repeating the Rp - Rz - Rp - Rz gates."""

        # Map unwanted Z rotations to qubit population.
        circ = QuantumCircuit(1)
        circ.rz(-np.pi / 2, 0)
        circ.sx(0)
        return circ

    def circuits(self) -> List[QuantumCircuit]:
        """Create the circuits for the fine DRAG calibration experiment.

        Returns:
            A list of circuits with a variable number of gates. Each gate has the same
            pulse schedule.
        """
        schedule, circuits = self.experiment_options.schedule, []

        for repetition in self.experiment_options.repetitions:
            circuit = self._pre_circuit()

            for _ in range(repetition):
                circuit.append(self.experiment_options.gate, (0,))
                circuit.rz(np.pi, 0)
                circuit.append(self.experiment_options.gate, (0,))
                circuit.rz(np.pi, 0)

            circuit.compose(self._post_circuit(), inplace=True)

            circuit.measure_all()

            if schedule is not None:
                circuit.add_calibration(
                    self.experiment_options.gate.name,
                    self.physical_qubits,
                    schedule,
                    params=[],
                )

            circuit.metadata = {"xval": repetition}

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


class FineXDrag(FineDrag):
    """Class to fine characterize the DRAG parameter of an X gate.

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=False, seed=199)

        .. jupyter-execute::

            from qiskit_experiments.library.characterization import FineXDrag

            exp = FineXDrag(physical_qubits=(0,), backend=backend)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)
    """

    def __init__(self, physical_qubits: Sequence[int], backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__(physical_qubits, XGate(), backend=backend)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the FineXDrag experiment.

        Experiment Options:
            gate (Gate): FineXDrag calibrates an XGate.
        """
        options = super()._default_experiment_options()
        options.gate = XGate()

        return options

    @staticmethod
    def _pre_circuit() -> QuantumCircuit:
        """Return the quantum circuit done before the Rp - Rz - Rp - Rz gates."""
        return QuantumCircuit(1)


class FineSXDrag(FineDrag):
    """Class to fine characterize the DRAG parameter of an :math:`SX` gate.

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
            backend = SingleTransmonTestBackend(5.2e9,-.25e9, 1e9, 0.8e9, 1e4, noise=False, seed=199)

        .. jupyter-execute::

            import numpy as np
            from qiskit_experiments.library.characterization import FineSXDrag

            exp = FineSXDrag(physical_qubits=(0,), backend=backend)
            exp.analysis.set_options(normalization= True,
                                     fixed_parameters={
                                         "angle_per_gate" : 0.0,
                                         "phase_offset" : np.pi/2,
                                         "amp" : 0.6
                                         },
                                    )

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)
    """

    def __init__(self, physical_qubits: Sequence[int], backend: Optional[Backend] = None):
        """Initialize the experiment."""
        super().__init__(physical_qubits, SXGate(), backend=backend)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the FineSXDrag experiment.

        Experiment Options:
            gate (Gate): FineSXDrag calibrates an SXGate.
        """
        options = super()._default_experiment_options()
        options.gate = SXGate()

        return options

    @staticmethod
    def _pre_circuit() -> QuantumCircuit:
        """Return the quantum circuit with an sx gate before the Rp - Rz - Rp - Rz gates."""
        circ = QuantumCircuit(1)
        circ.sx(0)
        return circ
