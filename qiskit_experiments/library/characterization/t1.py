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
T1 Experiment class.
"""

from typing import List, Optional, Union, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BackendTiming, BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis.t1_analysis import T1Analysis


class T1(BaseExperiment):
    r"""An experiment to measure the qubit relaxation time.

    # section: overview

        This experiment estimates the :math:`T_1` relaxation time of the qubit by
        generating a series of circuits that excite the qubit then wait for different
        intervals before measurement. The resulting data of excited population versus
        wait time is fitted to an exponential curve to obtain an estimate for
        :math:`T_1`.

    # section: analysis_ref
        :class:`.T1Analysis`

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_ibm_runtime.fake_provider import FakeManilaV2
            from qiskit_aer import AerSimulator
            backend = AerSimulator.from_backend(FakeManilaV2())

        .. jupyter-execute::

            import numpy as np
            from qiskit_experiments.library import T1

            delays = np.arange(1.e-6, 300.e-6, 30.e-6)
            exp = T1(physical_qubits=(0, ), delays=delays, backend=backend)

            exp_data = exp.run().block_for_results()
            display(exp_data.figure(0))
            exp_data.analysis_results(dataframe=True)

    # section: manual
        :doc:`/manuals/characterization/t1`

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments in seconds.
        """
        options = super()._default_experiment_options()
        options.delays = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Union[List[float], np.array],
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the T1 experiment class.

        Args:
            physical_qubits: a single-element sequence containing the qubit whose T1 is to be
                estimated.
            delays: Delay times of the experiments in seconds.
            backend: Optional, the backend to run the experiment on.

        Raises:
            ValueError: If the number of delays is smaller than 3
        """
        # Initialize base experiment
        super().__init__(physical_qubits, analysis=T1Analysis(), backend=backend)

        # Set experiment options
        self.set_experiment_options(delays=delays)

    def circuits(self, qnum=0) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Returns:
            The experiment circuits
        """
        timing = BackendTiming(self.backend)

        circuits = []
        for delay in self.experiment_options.delays:
            circ = QuantumCircuit(qnum+1, 1)
            circ.x(qnum)
            circ.barrier(qnum)
            circ.delay(timing.round_delay(time=delay), qnum, timing.delay_unit)
            circ.barrier(qnum)
            circ.measure(qnum, 0)

            circ.metadata = {"xval": timing.delay_time(time=delay)}

            circuits.append(circ)

        return circuits
    
    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled.

        Override to skip transpilaion if not needed
        """
        if self._backend:
            #get basis gates
            basis_gates = self._backend.configuration().basis_gates
            
            if 'x' in basis_gates:
                return self.circuits(self.physical_qubits[0])
            
        return super()._transpiled_circuits()


    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
