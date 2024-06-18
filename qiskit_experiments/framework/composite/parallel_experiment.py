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
Parallel Experiment class.
"""
from typing import List, Optional
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Clbit, Delay
from qiskit.providers.backend import Backend
from qiskit_experiments.exceptions import QiskitError
from .composite_experiment import CompositeExperiment, BaseExperiment
from .composite_analysis import CompositeAnalysis


class ParallelExperiment(CompositeExperiment):
    """Combine multiple experiments into a parallel experiment.

    Parallel experiments combine individual experiments on disjoint subsets
    of qubits into a single composite experiment on the union of those qubits.
    The component experiment circuits are combined to run in parallel on the
    respective qubits.

    Analysis of parallel experiments is performed using the
    :class:`~qiskit_experiments.framework.CompositeAnalysis` class which handles
    marginalizing the composite experiment circuit data into individual child
    :class:`ExperimentData` containers for each component experiment which are
    then analyzed using the corresponding analysis class for that component
    experiment.

    See :class:`~qiskit_experiments.framework.CompositeAnalysis`
    documentation for additional information.
    """

    def __init__(
        self,
        experiments: List[BaseExperiment],
        backend: Optional[Backend] = None,
        flatten_results: bool = True,
        analysis: Optional[CompositeAnalysis] = None,
        experiment_type: Optional[str] = None,
    ):
        """Initialize the analysis object.

        Args:
            experiments: a list of experiments.
            backend: Optional, the backend to run the experiment on.
            flatten_results: If True flatten all component experiment results
                             into a single ExperimentData container, including
                             nested composite experiments. If False save each
                             component experiment results as a separate child
                             ExperimentData container. This kwarg is ignored
                             if the analysis kwarg is used.
            analysis: Optional, the composite analysis class to use. If not
                      provided this will be initialized automatically from the
                      supplied experiments.
        """
        qubits = []
        for exp in experiments:
            qubits += exp.physical_qubits
        super().__init__(
            experiments,
            qubits,
            backend=backend,
            analysis=analysis,
            flatten_results=flatten_results,
            experiment_type=experiment_type,
        )

    def circuits(self):
        return self._combined_circuits(device_layout=False)

    def _transpiled_circuits(self):
        return self._combined_circuits(device_layout=True)

    def _combined_circuits(self, device_layout: bool) -> List[QuantumCircuit]:
        """Generate combined parallel circuits from transpiled subcircuits."""
        if not device_layout:
            # Num qubits will be computed from sub experiments
            num_qubits = len(self.physical_qubits)
        else:
            # Work around for backend coupling map circuit inflation
            coupling_map = getattr(self.transpile_options, "coupling_map", None)
            if coupling_map is None and self.backend:
                coupling_map = self._backend_data.coupling_map
            if coupling_map is not None:
                num_qubits = 1 + max(*self.physical_qubits, np.max(coupling_map))
            else:
                num_qubits = 1 + max(self.physical_qubits)

        # Transpile component circuits
        transpiled_circuits = []

        # Max duration for all components that will be combined into a single circuit
        max_durations = {}
        duration_unit = None
        scheduling_method = getattr(self.transpile_options, "scheduling_method", None)

        # Find max durations of subcircuits
        for exp_idx, sub_exp in enumerate(self._experiments):
            # Generate transpiled subcircuits
            sub_circuits = sub_exp._transpiled_circuits()
            transpiled_circuits.append(sub_circuits)
            if scheduling_method is not None:
                for circ_idx, sub_circ in enumerate(sub_circuits):
                    if sub_circ.duration is not None:
                        if duration_unit is None:
                            duration_unit = sub_circ.unit
                        if circ_idx not in max_durations:
                            max_durations[circ_idx] = 0
                        max_durations[circ_idx] = max(sub_circ.duration, max_durations[circ_idx])
                        if duration_unit != sub_circ.unit:
                            raise QiskitError(
                                "Scheduled component experiments are scheduled with"
                                " different time units."
                            )

        # Combine circuits
        joint_circuits = []
        sub_qubits = 0
        for exp_idx, (sub_circuits, sub_exp) in enumerate(
            zip(transpiled_circuits, self._experiments)
        ):
            # Qubit remapping for non-transpiled circuits
            if not device_layout:
                qubits = list(range(sub_qubits, sub_qubits + sub_exp.num_qubits))
                qargs_map = {q: qubits[i] for i, q in enumerate(sub_exp.physical_qubits)}
                sub_qubits += sub_exp.num_qubits
            else:
                qubits = list(sub_exp.physical_qubits)
                qargs_map = {q: q for q in sub_exp.physical_qubits}

            for circ_idx, sub_circ in enumerate(sub_circuits):
                if circ_idx >= len(joint_circuits):
                    # Initialize new joint circuit or extract
                    # existing circuit if already initialized
                    new_circuit = QuantumCircuit(num_qubits, name=f"parallel_exp_{circ_idx}")
                    new_circuit.metadata = {
                        "experiment_type": self._type,
                        "composite_index": [],
                        "composite_metadata": [],
                        "composite_qubits": [],
                        "composite_clbits": [],
                    }
                    joint_circuits.append(new_circuit)

                # Add classical registers required by subcircuit
                circuit = joint_circuits[circ_idx]
                num_clbits = circuit.num_clbits
                sub_clbits = sub_circ.num_clbits
                clbits = list(range(num_clbits, num_clbits + sub_clbits))
                if sub_clbits:
                    creg = ClassicalRegister(sub_clbits)
                    sub_cargs = [Clbit(creg, i) for i in range(sub_clbits)]
                    circuit.add_register(creg)
                else:
                    sub_cargs = []

                # Apply transpiled subcircuit
                # Note that this assumes the circuit was not expanded to use
                # any qubits outside the specified physical qubits
                circ_duration = max_durations.get(circ_idx)
                pad_time = None
                if scheduling_method and sub_circ.duration and circ_duration:
                    pad_time = abs(circ_duration - sub_circ.duration)

                # If scheduling method is alap prepend shorter sub-circuits with delays
                if scheduling_method == "alap" and pad_time:
                    for i in sub_exp.physical_qubits:
                        circuit._append(
                            Delay(pad_time, unit=duration_unit), [circuit.qubits[i]], []
                        )

                for inst, qargs, cargs in sub_circ.data:
                    mapped_cargs = [sub_cargs[sub_circ.find_bit(i).index] for i in cargs]
                    try:
                        mapped_qargs = [
                            circuit.qubits[qargs_map[sub_circ.find_bit(i).index]] for i in qargs
                        ]
                    except KeyError as ex:
                        # Instruction is outside physical qubits for the component
                        # experiment.
                        # This could legitimately happen if the subcircuit was
                        # explicitly scheduled during transpilation which would
                        # insert delays on all auxillary device qubits.
                        # We skip delay instructions to allow for this.
                        if inst.name == "delay":
                            continue
                        raise QiskitError(
                            "Component experiment has been transpiled outside of the "
                            "allowed physical qubits for that component. Check the "
                            "experiment is valid on the backends coupling map."
                        ) from ex
                    circuit._append(inst, mapped_qargs, mapped_cargs)

                # If scheduling method is alap append shorter sub-circuits with delays
                if scheduling_method == "asap" and pad_time:
                    for i in sub_exp.physical_qubits:
                        circuit._append(
                            Delay(pad_time, unit=duration_unit), [circuit.qubits[i]], []
                        )

                # Update duration of circuit
                if scheduling_method and circ_duration:
                    circuit.duration = circ_duration
                    circuit.unit = duration_unit

                # Add subcircuit metadata
                circuit.metadata["composite_index"].append(exp_idx)
                circuit.metadata["composite_metadata"].append(sub_circ.metadata)
                circuit.metadata["composite_qubits"].append(qubits)
                circuit.metadata["composite_clbits"].append(clbits)

                # Add the calibrations
                for gate, cals in sub_circ.calibrations.items():
                    for key, sched in cals.items():
                        circuit.add_calibration(gate, qubits=key[0], schedule=sched, params=key[1])

        return joint_circuits
