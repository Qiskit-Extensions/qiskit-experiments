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
Standard GST Experiment class.
"""
from typing import Union, Optional, List, Dict, Callable, Tuple
from types import FunctionType
import copy
import itertools as itert
from qiskit import QuantumCircuit, QiskitError
from qiskit.providers import Backend
from qiskit.circuit import Gate

from qiskit_experiments.framework import BaseExperiment, Options
from .gst_analysis import GSTAnalysis
from .gatesetbasis import (
    GateSetBasis,
    default_gateset_basis,
    gatesetbasis_constrction,
)


class GateSetTomography(BaseExperiment):
    r"""Gate set tomography experiment.

    # section: overview

        Gate-Set Tomography is a powerful method that, similar
        to Quantum Process Tomography, is used to perform full
        characterization of quantum processes using measurement data of a specific set of quantum
        circuits.
        Unlike the quantum process tomography, the gate set tomography deals with
        the state preparation and measurements (SPAM) errors self-consistently by
        including the gates used for both initializing and measuring the qubits in
        the gate set which is reconstructed from processing measurement data.
        Therefore, it does not need any pre-calibration of the initialized qubits
        or the measurements and provides full characterization of the states,
        quantum processes and measurements.

        Gate set tomography is performed on a gate set :math:`G = (G0, G1,...,Gm)`
        with the additional information of SPAM circuits :math:`F = (F0,F1,...,Fn)`
        that are constructed from the gates in the gate set.
        In gate set tomography, we assume a single initial state rho which is
        the native state in which all states are initialized in 0 state
        and a single POVM measurement operator E, a measurement of all qubits
        in state 0. The SPAM circuits provide us with a complete set of initial
        state F_j|rho> and measurements <E|F_i.
        We perform three types of experiments:
        1) :math:`\langle E  | F_i G_k F_j |\rho \rangle`
        for 1 <= i,j <= n and 1 <= k <= m:
            This experiment enables us to obtain data on the gate G_k
        2) :math:`\langle E  | F_i F_j |\rho \rangle`  for 1 <= i,j <= n:
            This experiment enables us to obtain the Gram matrix required
            to "invert" the results of experiments of type 1 in order to
            reconstruct (a matrix similar to) the gate G_k
        3) :math:`\langle E  | F_j |\rho \rangle` for 1 <= j <= n:
            This experiment enables us to reconstruct <E| and rho

        This experiment protocol is based on the protocol described in section 3.5.2 in
        arXiv:1509.02921.

    # section: reference
        .. ref_arxiv:: 1 1509.02921
    """

    # Analysis class for experiment
    __analysis_class__ = GSTAnalysis

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()

        return options

    def __init__(
        self,
        qubits: List[int],
        basis_gates: Union[str, Dict[str, FunctionType]] = "default",
        spam_gates: Union[str, Dict[str, Tuple]] = "default",
        additional_gates: Dict[str, Union[Callable, Gate]] = None,
        only_basis_gates: bool = False,
    ):
        """Initialize a gate set tomography experiment.

        Args:
            qubits: A List of qubits labels GST is performed on.
            basis_gates: A dictionary of the basis gates set :math:`G`, provided as in the following
                example:
                    basis_gates = {
                    'Id': lambda circ, qubit: None,
                    'X_Rot_90': lambda circ, qubit: circ.append(U2Gate(-np.pi / 2, np.pi / 2), [qubit]),
                    'Y_Rot_90': lambda circ, qubit: circ.append(U2Gate(0, 0), [qubit])
                    }
                where the keys are the labels of the gates, and the values are function which takes
                a QuantumCircuit, and QuantumRegister (or a list of them) as arguments and returns
                a description of the gate action on the qubits.

                If 'default' or None, the built-in default gate set will be used.

            spam_gates: A dictionary of SPAM gates :math:`F`, where the keys are spam gates label of the
                form 'F'+'spam gate number', and the values are tuples of the basis gates labels
                that construct the spam gate. For example:
                    spam_gates = {
                    'F0': ('Id',),
                    'F1': ('X_Rot_90',),
                    'F2': ('Y_Rot_90',),
                    'F3': ('X_Rot_90', 'X_Rot_90')
                    }
                If 'default' or None, the built-in default gate set will be used if only_basis_gates
                is False. Otherwise, if only_basis_gates below is true, a built_in function will be used
                to construct spam gates.
            additional_gates: The list of gates to add to the `basis_gates` set to be characterized
            using GST.
            only_basis_gates: A boolean variable indicating whether spam gates are provided or
                only a set of basis gates is provided from which the algorithm is required to
                construct a set of SPAM gates.
        Raises:
            QiskitError: If the provided `basis_gates` and `spam_gates` are of a wrong
            format.
        """

        # Initialize base experiment
        super().__init__(qubits)

        if only_basis_gates is False:
            if isinstance(basis_gates, dict):
                if spam_gates == "default":
                    raise QiskitError(
                        "Spam gates are not provided. "
                        "If only basis_gates are provided, only_basis_gates should be True."
                    )

                try:
                    self._gateset_basis = GateSetBasis(
                        "Gate Set", basis_gates, spam_gates, self.num_qubits
                    )

                except (ValueError, QiskitError) as gateset_wrong_format:
                    raise QiskitError(
                        "Can not run GST experiment with the provided gate set format."
                    ) from gateset_wrong_format

            elif basis_gates == "default":
                if self.num_qubits < 3:
                    self._gateset_basis = default_gateset_basis(self.num_qubits)
                else:
                    raise QiskitError(
                        "Only 1-qubit and 2-qubit default gate sets are available" " in Qiskit"
                    )

        elif only_basis_gates is True and isinstance(basis_gates, dict):
            self._gateset_basis = gatesetbasis_constrction(basis_gates, self.num_qubits)[0]
        else:
            raise QiskitError("Can not run GST experiment with the provided gate set format.")

        if additional_gates is not None:
            for gate_label in additional_gates.keys():
                self._gateset_basis.add_gate(additional_gates[gate_label], gate_label)

    def _metadata(self):
        metadata = super()._metadata()
        # A dding measured_qubits and gateset basis to metadata
        if self.physical_qubits:
            metadata["measured_qubits"] = copy.copy(self.physical_qubits)
        if self._gateset_basis:
            metadata["gateset_basis"] = copy.copy(self._gateset_basis)
        return metadata

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        r"""Return a list of GST circuits.

        Args:
            backend (Backend): Optional, a backend object.

        Returns:
            A list of :class:`QuantumCircuit`.

        Additional Information:
            It returns the circuits corresponding to measuring
            :math:`\langle E  | F_i G_k F_j |\rho \rangle`,
            :math:`\langle E  | F_i F_j |\rho \rangle` and :math:`\langle E  | F_j |\rho \rangle`
            as explained in further details in the overview of GateSetTomography class.

        """
        all_circuits = []

        # Experiments of the form <E|F_i G_k F_j|rho>
        fgf_circuits = []
        for gate in self._gateset_basis.gate_labels:
            for (fprep, fmeas) in itert.product(
                self._gateset_basis.spam_labels, self._gateset_basis.spam_labels
            ):
                circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
                self._gateset_basis.add_spam_to_circuit(circuit, self.physical_qubits, fprep)
                circuit.barrier()
                self._gateset_basis.add_gate_to_circuit(circuit, self.physical_qubits, gate)
                circuit.barrier()
                self._gateset_basis.add_spam_to_circuit(circuit, self.physical_qubits, fmeas)
                circuit.measure(self.physical_qubits, self.physical_qubits)
                circuit.name = str((fprep, gate, fmeas))
                metadata = {
                    "experiment_type": self._type,
                    "circuit_name": circuit.name,
                }
                circuit.metadata = metadata
                fgf_circuits.append(circuit)

        all_circuits = all_circuits + fgf_circuits

        # Experiments of the form <E|F_i F_j|rho>
        # Can be skipped if one of the gates is ideal identity

        ff_circuits = []
        for (fprep, fmeas) in itert.product(
            self._gateset_basis.spam_labels, self._gateset_basis.spam_labels
        ):
            circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
            self._gateset_basis.add_spam_to_circuit(circuit, self.physical_qubits, fprep)
            circuit.barrier()
            self._gateset_basis.add_spam_to_circuit(circuit, self.physical_qubits, fmeas)
            circuit.measure(self.physical_qubits, self.physical_qubits)
            circuit.name = str((fprep, fmeas))
            metadata = {
                "experiment_type": self._type,
                "circuit_name": circuit.name,
            }
            circuit.metadata = metadata
            ff_circuits.append(circuit)
        all_circuits = all_circuits + ff_circuits

        # Experiments of the form <E|F_j|rho>

        f_circuits = []
        for fprep in self._gateset_basis.spam_labels:
            circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
            self._gateset_basis.add_spam_to_circuit(circuit, self.physical_qubits, fprep)
            circuit.barrier()
            circuit.measure(self.physical_qubits, self.physical_qubits)
            circuit.name = str((fprep,))
            metadata = {
                "experiment_type": self._type,
                "circuit_name": circuit.name,
            }
            circuit.metadata = metadata
            f_circuits.append(circuit)
        all_circuits = all_circuits + f_circuits

        return all_circuits
