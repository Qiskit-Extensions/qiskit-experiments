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
"""
T2Hahn Echo Experiment class.
"""

from typing import List, Optional, Union, Sequence
import numpy as np

from qiskit import QuantumCircuit, QiskitError
from qiskit.circuit import Parameter
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BackendTiming, BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis.t2hahn_analysis import T2HahnAnalysis


class T2Hahn(BaseExperiment):
    r"""An experiment to measure the dephasing time insensitive to inhomogeneous
    broadening using Hahn echos.

    # section: overview

        This experiment is used to estimate the :math:`T_2` time of a single qubit.
        :math:`T_2` is the dephasing time or the transverse relaxation time of the qubit
        on the Bloch sphere as a result of both energy relaxation and pure dephasing in
        the transverse plane. Unlike :math:`T_2^*`, which is measured by
        :class:`.T2Ramsey`, :math:`T_2` is insensitive to inhomogenous broadening.

        This experiment consists of a series of circuits of the form


        .. parsed-literal::

                 ┌─────────┐┌──────────┐┌───────┐┌──────────┐┌─────────┐┌─┐
            q_0: ┤ Rx(π/2) ├┤ DELAY(t) ├┤ RX(π) ├┤ DELAY(t) ├┤ RX(π/2) ├┤M├
                 └─────────┘└──────────┘└───────┘└──────────┘└─────────┘└╥┘
            c: 1/════════════════════════════════════════════════════════╩═
                                                                         0

        for each *t* from the specified delay times
        and the delays are specified by the user.
        The delays that are specified are delay for each delay gate while
        the delay in the metadata is the total delay which is delay * (num_echoes +1)
        The circuits are run on the device or on a simulator backend.

    # section: example

        Here's a minimal working example:

        .. jupyter-execute::
            :hide-code:

            from qiskit_experiments.test.t2hahn_backend import T2HahnBackend

            osc_freq = 0.1
            estimated_t2hahn = 20

            backend = T2HahnBackend(
                            t2hahn=[estimated_t2hahn],
                            frequency=[osc_freq],
                            initialization_error=[0.0],
                            readout0to1=[0.02],
                            readout1to0=[0.02],
                       )

        .. jupyter-execute::

            from qiskit_experiments.library import T2Hahn

            exp = T2Hahn(physical_qubits = (0,),
                         delays = [1e-6, 5e-6, 10e-6])

            exp_data = exp.run(backend).block_for_results()
            exp_data.figure(0)

    # section: manual
        :doc:`/manuals/characterization/t2hahn`

    # section: analysis_ref
        :class:`T2HahnAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1904.06560
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments.
            num_echoes (int): The number of echoes to preform.
        """
        options = super()._default_experiment_options()

        options.delays = None
        options.num_echoes = 1
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Union[List[float], np.array],
        num_echoes: int = 1,
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the T2 Hahn Echo class.

        Args:
            physical_qubits: a single-element sequence containing the qubit whose T2 is to be
                estimated
            delays: Total delay times of the experiments.
            backend: Optional, the backend to run the experiment on.
            num_echoes: The number of echoes to preform.
            backend: Optional, the backend to run the experiment on.

         Raises:
             QiskitError : Error for invalid input.
        """
        # Initialize base experiment
        super().__init__(physical_qubits, analysis=T2HahnAnalysis(), backend=backend)

        # Set experiment options
        self.set_experiment_options(delays=delays, num_echoes=num_echoes)
        self._verify_parameters()

    def _verify_parameters(self):
        """
        Verify input correctness, raise QiskitError if needed.

        Raises:
            QiskitError : Error for invalid input.
        """
        if any(delay < 0 for delay in self.experiment_options.delays):
            raise QiskitError(
                f"The lengths list {self.experiment_options.delays} should only contain "
                "non-negative elements."
            )

    def circuits(self) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits.

        Each circuit consists of RX(π/2) followed by a sequence of delay gate,
        RX(π) for echo and delay gate again.
        The sequence repeats for the number of echoes and terminates with RX(±π/2).

        Returns:
            The experiment circuits.
        """
        timing = BackendTiming(self.backend)

        template = QuantumCircuit(1, 1)
        template.metadata = {
            "experiment_type": self._type,
            "qubit": self.physical_qubits[0],
            "unit": "s",
        }

        delay_param = Parameter("delay")

        num_echoes = self.experiment_options.num_echoes

        # First X rotation in 90 degrees
        template.rx(np.pi / 2, 0)  # Brings the qubit to the X Axis
        if num_echoes == 0:
            # if number of echoes is 0 then just apply the delay gate
            template.delay(delay_param, 0, timing.delay_unit)
        else:
            for _ in range(num_echoes):
                template.delay(delay_param, 0, timing.delay_unit)
                template.rx(np.pi, 0)
                template.delay(delay_param, 0, timing.delay_unit)

        if num_echoes % 2 == 1:
            template.rx(np.pi / 2, 0)  # X90 again since the num of echoes is odd
        else:
            template.rx(-np.pi / 2, 0)  # X(-90) again since the num of echoes is even
        template.measure(0, 0)  # measure

        circuits = []
        for delay in self.experiment_options.delays:
            if num_echoes == 0:
                single_delay = timing.delay_time(time=delay)
                total_delay = single_delay
            else:
                # Equal delay is put before and after each echo, so each echo gets
                # two delay gates. When there are multiple echoes, the total delay
                # between echoes is 2 * single_delay, made up of two delay gates.
                single_delay = timing.delay_time(time=delay / num_echoes / 2)
                total_delay = single_delay * num_echoes * 2

            assigned = template.assign_parameters(
                {delay_param: timing.round_delay(time=single_delay)}, inplace=False
            )
            assigned.metadata["xval"] = total_delay
            circuits.append(assigned)

        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
