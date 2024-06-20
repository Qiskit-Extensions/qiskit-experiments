# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Utilities for using the Clifford group in randomized benchmarking.
"""

import itertools
import os
from functools import lru_cache
from numbers import Integral
from typing import Optional, Union, Tuple, Sequence, Iterable

import numpy as np

from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.circuit import Gate, Instruction
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import SdgGate, HGate, SGate, XGate, YGate, ZGate
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig, HighLevelSynthesis

DEFAULT_SYNTHESIS_METHOD = "rb_default"

_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")

_CLIFFORD_COMPOSE_1Q = np.load(f"{_DATA_FOLDER}/clifford_compose_1q.npz")["table"]
_CLIFFORD_INVERSE_1Q = np.load(f"{_DATA_FOLDER}/clifford_inverse_1q.npz")["table"]
_CLIFFORD_INVERSE_2Q = np.load(f"{_DATA_FOLDER}/clifford_inverse_2q.npz")["table"]
_clifford_compose_2q_data = np.load(f"{_DATA_FOLDER}/clifford_compose_2q_dense_selected.npz")
_CLIFFORD_COMPOSE_2Q_DENSE = _clifford_compose_2q_data["table"]
# valid indices for the columns of the _CLIFFORD_COMPOSE_2Q_DENSE table
_valid_sparse_indices = _clifford_compose_2q_data["valid_sparse_indices"]
# map a clifford number to the index of _CLIFFORD_COMPOSE_2Q_DENSE
_clifford_num_to_dense_index = {idx: ii for ii, idx in enumerate(_valid_sparse_indices)}
_CLIFFORD_TENSOR_1Q = np.load(f"{_DATA_FOLDER}/clifford_tensor_1q.npz")["table"]


# Transpilation utilities
def _transpile_clifford_circuit(
    circuit: QuantumCircuit, physical_qubits: Sequence[int]
) -> QuantumCircuit:
    # Simplified transpile that only decomposes Clifford circuits and creates the layout.
    return _apply_qubit_layout(_decompose_clifford_ops(circuit), physical_qubits=physical_qubits)


def _decompose_clifford_ops(circuit: QuantumCircuit) -> QuantumCircuit:
    # Simplified QuantumCircuit.decompose, which decomposes only Clifford ops
    res = circuit.copy_empty_like()
    if hasattr(circuit, "_parameter_table"):
        res._parameter_table = circuit._parameter_table
    for inst in circuit:
        if inst.operation.name.startswith("Clifford"):  # Decompose
            rule = inst.operation.definition.data
            if len(rule) == 1 and len(inst.qubits) == len(rule[0].qubits):
                if inst.operation.definition.global_phase:
                    res.global_phase += inst.operation.definition.global_phase
                res._data.append(
                    CircuitInstruction(
                        operation=rule[0].operation,
                        qubits=inst.qubits,
                        clbits=inst.clbits,
                    )
                )
            else:
                _circuit_compose(res, inst.operation.definition, qubits=inst.qubits)
        else:  # Keep the original instruction
            res._data.append(inst)
    return res


def _apply_qubit_layout(circuit: QuantumCircuit, physical_qubits: Sequence[int]) -> QuantumCircuit:
    # Mapping qubits in circuit to physical qubits (layout)
    res = QuantumCircuit(1 + max(physical_qubits), name=circuit.name, metadata=circuit.metadata)
    res.add_bits(circuit.clbits)
    for reg in circuit.cregs:
        res.add_register(reg)
    _circuit_compose(res, circuit, qubits=physical_qubits)
    if hasattr(circuit, "_parameter_table"):
        res._parameter_table = circuit._parameter_table
    return res


def _circuit_compose(
    self: QuantumCircuit, other: QuantumCircuit, qubits: Sequence[Union[Qubit, int]]
) -> QuantumCircuit:
    # Simplified QuantumCircuit.compose with clbits=None, front=False, inplace=True, wrap=False
    # without any validation, parameter_table/calibrations updates and copy of operations
    # The input circuit `self` is changed inplace.
    qubit_map = {
        other.qubits[i]: (self.qubits[q] if isinstance(q, int) else q) for i, q in enumerate(qubits)
    }
    for instr in other:
        self._data.append(
            CircuitInstruction(
                operation=instr.operation,
                qubits=[qubit_map[q] for q in instr.qubits],
                clbits=instr.clbits,
            ),
        )
    self.global_phase += other.global_phase
    return self


def _synthesize_clifford(
    clifford: Clifford,
    basis_gates: Optional[Tuple[str]],
    coupling_tuple: Optional[Tuple[Tuple[int, int]]] = None,
    synthesis_method: str = DEFAULT_SYNTHESIS_METHOD,
) -> QuantumCircuit:
    """Synthesize a circuit of a Clifford element. The resulting circuit contains only
    ``basis_gates`` and it complies with ``coupling_tuple``.

    Args:
        clifford: Clifford element to be converted
        basis_gates: basis gates to use in the conversion
        coupling_tuple: coupling map to use in the conversion in the form of tuple of edges
        synthesis_method: conversion algorithm name

    Returns:
        Synthesized circuit
    """
    qc = QuantumCircuit(clifford.num_qubits, name=str(clifford))
    qc.append(clifford, qc.qubits)
    return _synthesize_clifford_circuit(
        qc,
        basis_gates=basis_gates,
        coupling_tuple=coupling_tuple,
        synthesis_method=synthesis_method,
    )


def _synthesize_clifford_circuit(
    circuit: QuantumCircuit,
    basis_gates: Optional[Tuple[str]],
    coupling_tuple: Optional[Tuple[Tuple[int, int]]] = None,
    synthesis_method: str = DEFAULT_SYNTHESIS_METHOD,
) -> QuantumCircuit:
    """Convert a Clifford circuit into one composed of ``basis_gates`` with
    satisfying ``coupling_tuple`` using the specified synthesis method.

    Args:
        circuit: Clifford circuit to be converted
        basis_gates: basis gates to use in the conversion
        coupling_tuple: coupling map to use in the conversion in the form of tuple of edges
        synthesis_method: name of Clifford synthesis algorithm to use

    Returns:
        Synthesized circuit
    """
    if basis_gates:
        basis_gates = list(basis_gates)
    coupling_map = CouplingMap(coupling_tuple) if coupling_tuple else None

    # special handling for 1q or 2q case for speed
    if circuit.num_qubits <= 2:
        if synthesis_method == DEFAULT_SYNTHESIS_METHOD:
            return transpile(
                circuit,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                optimization_level=1,
            )
        else:
            # Provided custom synthesis method, re-synthesize Clifford circuit
            # convert the circuit back to a Clifford object and then call the synthesis plugin
            new_circuit = QuantumCircuit(circuit.num_qubits, name=circuit.name)
            new_circuit.append(Clifford(circuit), new_circuit.qubits)
            circuit = new_circuit

    # for 3q+ or custom synthesis method, synthesizes clifford circuit
    hls_config = HLSConfig(clifford=[(synthesis_method, {"basis_gates": basis_gates})])
    pm = PassManager([HighLevelSynthesis(hls_config=hls_config, coupling_map=coupling_map)])
    circuit = pm.run(circuit)
    return circuit


@lru_cache(maxsize=256)
def _clifford_1q_int_to_instruction(
    num: Integral,
    basis_gates: Optional[Tuple[str]],
    synthesis_method: str = DEFAULT_SYNTHESIS_METHOD,
) -> Instruction:
    return CliffordUtils.clifford_1_qubit_circuit(
        num, basis_gates=basis_gates, synthesis_method=synthesis_method
    ).to_instruction()


@lru_cache(maxsize=11520)
def _clifford_2q_int_to_instruction(
    num: Integral,
    basis_gates: Optional[Tuple[str]],
    coupling_tuple: Optional[Tuple[Tuple[int, int]]],
    synthesis_method: str = DEFAULT_SYNTHESIS_METHOD,
) -> Instruction:
    return CliffordUtils.clifford_2_qubit_circuit(
        num,
        basis_gates=basis_gates,
        coupling_tuple=coupling_tuple,
        synthesis_method=synthesis_method,
    ).to_instruction()


def _hash_cliff(cliff):
    return cliff.tableau.tobytes(), cliff.tableau.shape


def _dehash_cliff(cliff_hash):
    tableau = np.frombuffer(cliff_hash[0], dtype=bool).reshape(cliff_hash[1])
    return Clifford(tableau)


def _clifford_to_instruction(
    clifford: Clifford,
    basis_gates: Optional[Tuple[str]],
    coupling_tuple: Optional[Tuple[Tuple[int, int]]],
    synthesis_method: str = DEFAULT_SYNTHESIS_METHOD,
) -> Instruction:
    return _cached_clifford_to_instruction(
        _hash_cliff(clifford),
        basis_gates=basis_gates,
        coupling_tuple=coupling_tuple,
        synthesis_method=synthesis_method,
    )


@lru_cache(maxsize=256)
def _cached_clifford_to_instruction(
    cliff_hash: Tuple[str, Tuple[int, int]],
    basis_gates: Optional[Tuple[str]],
    coupling_tuple: Optional[Tuple[Tuple[int, int]]],
    synthesis_method: str = DEFAULT_SYNTHESIS_METHOD,
) -> Instruction:
    return _synthesize_clifford(
        _dehash_cliff(cliff_hash),
        basis_gates=basis_gates,
        coupling_tuple=coupling_tuple,
        synthesis_method=synthesis_method,
    ).to_instruction()


# The classes VGate and WGate are not actually used in the code - we leave them here to give
# a better understanding of the composition of the layers for 2-qubit Cliffords.
class VGate(Gate):
    """V Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new V Gate."""
        super().__init__("v", 1, [])

    def _define(self):
        """V Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(SdgGate(), [q[0]], []), (HGate(), [q[0]], [])]
        self.definition = qc


class WGate(Gate):
    """W Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new W Gate."""
        super().__init__("w", 1, [])

    def _define(self):
        """W Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(HGate(), [q[0]], []), (SGate(), [q[0]], [])]
        self.definition = qc


class CliffordUtils:
    """Utilities for generating one- and two-qubit Clifford circuits and elements."""

    NUM_CLIFFORD_1_QUBIT = 24
    NUM_CLIFFORD_2_QUBIT = 11520
    CLIFFORD_1_QUBIT_SIG = (2, 3, 4)
    CLIFFORD_2_QUBIT_SIGS = [  # TODO: deprecate
        (2, 2, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 4, 4),
    ]

    @classmethod
    @lru_cache(maxsize=24)
    def clifford_1_qubit(cls, num):
        """Return the 1-qubit clifford element corresponding to `num`
        where `num` is between 0 and 23.
        """
        return Clifford(cls.clifford_1_qubit_circuit(num), validate=False)

    @classmethod
    @lru_cache(maxsize=11520)
    def clifford_2_qubit(cls, num):
        """Return the 2-qubit clifford element corresponding to ``num``,
        where ``num`` is between 0 and 11519.
        """
        return Clifford(cls.clifford_2_qubit_circuit(num), validate=False)

    @classmethod
    @lru_cache(maxsize=24)
    def clifford_1_qubit_circuit(
        cls,
        num,
        basis_gates: Optional[Tuple[str, ...]] = None,
        synthesis_method: str = DEFAULT_SYNTHESIS_METHOD,
    ):
        """Return the 1-qubit clifford circuit corresponding to ``num``,
        where ``num`` is between 0 and 23.
        """
        unpacked = cls._unpack_num(num, cls.CLIFFORD_1_QUBIT_SIG)
        i, j, p = unpacked[0], unpacked[1], unpacked[2]
        qc = QuantumCircuit(1, name=f"Clifford-1Q({num})")
        if i == 1:
            qc.h(0)
        if j == 1:
            qc.sxdg(0)
        if j == 2:
            qc.s(0)
        if p == 1:
            qc.x(0)
        if p == 2:
            qc.y(0)
        if p == 3:
            qc.z(0)

        if basis_gates:
            qc = _synthesize_clifford_circuit(qc, basis_gates, synthesis_method=synthesis_method)

        return qc

    @classmethod
    @lru_cache(maxsize=11520)
    def clifford_2_qubit_circuit(
        cls,
        num,
        basis_gates: Optional[Tuple[str, ...]] = None,
        coupling_tuple: Optional[Tuple[Tuple[int, int]]] = None,
        synthesis_method: str = DEFAULT_SYNTHESIS_METHOD,
    ):
        """Return the 2-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 11519.
        """
        qc = QuantumCircuit(2, name=f"Clifford-2Q({num})")
        for layer, idx in enumerate(_layer_indices_from_num(num)):
            if basis_gates:
                layer_circ = _transformed_clifford_layer(
                    layer, idx, basis_gates, coupling_tuple, synthesis_method=synthesis_method
                )
            else:
                layer_circ = _CLIFFORD_LAYER[layer][idx]
            _circuit_compose(qc, layer_circ, qubits=(0, 1))

        return qc

    @staticmethod
    def _unpack_num(num, sig):
        r"""Returns a tuple :math:`(a_1, \ldots, a_n)` where
        :math:`0 \le a_i \le \sigma_i` where
        sig=:math:`(\sigma_1, \ldots, \sigma_n)` and num is the sequential
        number of the tuple
        """
        res = []
        for k in sig:
            res.append(num % k)
            num //= k
        return res


# Constant mapping from 1Q single Clifford gate to 1Q Clifford numerical identifier.
# This table must be generated using `data.generate_clifford_data.gen_cliff_single_1q_gate_map`, or,
# equivalently, correspond to the ordering implicitly defined by CliffUtils.clifford_1_qubit_circuit.
_CLIFF_SINGLE_GATE_MAP_1Q = {
    ("id", (0,)): 0,
    ("h", (0,)): 1,
    ("sxdg", (0,)): 2,
    ("s", (0,)): 4,
    ("x", (0,)): 6,
    ("sx", (0,)): 8,
    ("y", (0,)): 12,
    ("z", (0,)): 18,
    ("sdg", (0,)): 22,
}
# Constant mapping from 2Q single Clifford gate to 2Q Clifford numerical identifier.
# This table must be generated using `data.generate_clifford_data.gen_cliff_single_2q_gate_map`, or,
# equivalently, correspond to the ordering defined by _layer_indices_from_num and _CLIFFORD_LAYER.
_CLIFF_SINGLE_GATE_MAP_2Q = {
    ("id", (0,)): 0,
    ("id", (1,)): 0,
    ("h", (0,)): 5760,
    ("h", (1,)): 2880,
    ("sxdg", (0,)): 6720,
    ("sxdg", (1,)): 3200,
    ("s", (0,)): 7680,
    ("s", (1,)): 3520,
    ("x", (0,)): 4,
    ("x", (1,)): 1,
    ("sx", (0,)): 6724,
    ("sx", (1,)): 3201,
    ("y", (0,)): 8,
    ("y", (1,)): 2,
    ("z", (0,)): 12,
    ("z", (1,)): 3,
    ("sdg", (0,)): 7692,
    ("sdg", (1,)): 3523,
    ("cx", (0, 1)): 16,
    ("cx", (1, 0)): 2336,
    ("cz", (0, 1)): 368,
    ("cz", (1, 0)): 368,
}


########
# Functions for 1-qubit integer Clifford operations
def compose_1q(lhs: Integral, rhs: Integral) -> Integral:
    """Return the composition of 1-qubit clifford integers."""
    return _CLIFFORD_COMPOSE_1Q[lhs, rhs]


def inverse_1q(num: Integral) -> Integral:
    """Return the inverse of a 1-qubit clifford integer."""
    return _CLIFFORD_INVERSE_1Q[num]


def num_from_1q_circuit(qc: QuantumCircuit) -> Integral:
    """Convert a given 1-qubit Clifford circuit to the corresponding integer.

    Note: The circuit must consist of gates in :const:`_CLIFF_SINGLE_GATE_MAP_1Q`,
    RZGate, Delay and Barrier.
    """
    num = 0
    for inst in qc:
        rhs = _num_from_1q_gate(op=inst.operation)
        num = _CLIFFORD_COMPOSE_1Q[num, rhs]
    return num


def _num_from_1q_gate(op: Instruction) -> int:
    """
    Convert a given 1-qubit clifford operation to the corresponding integer.
    Note that supported operations are limited to ones in :const:`_CLIFF_SINGLE_GATE_MAP_1Q` or Rz gate.

    Args:
        op: operation to be converted.

    Returns:
        An integer representing a Clifford consisting of a single operation.

    Raises:
        QiskitError: If the input instruction is not a Clifford instruction.
        QiskitError: If rz is given with a angle that is not Clifford.
    """
    if op.name in {"delay", "barrier"}:
        return 0
    try:
        name = _deparameterized_name(op)
        return _CLIFF_SINGLE_GATE_MAP_1Q[(name, (0,))]
    except QiskitError as err:
        raise QiskitError(
            f"Parameterized instruction {op.name} could not be converted to integer Clifford"
        ) from err
    except KeyError as err:
        raise QiskitError(
            f"Instruction {op.name} could not be converted to integer Clifford"
        ) from err


def _deparameterized_name(inst: Instruction) -> str:
    if inst.name == "rz":
        if np.isclose(inst.params[0], np.pi) or np.isclose(inst.params[0], -np.pi):
            return "z"
        elif np.isclose(inst.params[0], np.pi / 2):
            return "s"
        elif np.isclose(inst.params[0], -np.pi / 2):
            return "sdg"
        else:
            raise QiskitError(f"Wrong param {inst.params[0]} for rz in clifford")

    return inst.name


########
# Functions for 2-qubit integer Clifford operations
def compose_2q(lhs: Integral, rhs: Integral) -> Integral:
    """Return the composition of 2-qubit clifford integers."""
    num = lhs
    for layer, idx in enumerate(_layer_indices_from_num(rhs)):
        gate_numbers = _CLIFFORD_LAYER_NUMS[layer][idx]
        for n in gate_numbers:
            num = _CLIFFORD_COMPOSE_2Q_DENSE[num, _clifford_num_to_dense_index[n]]
    return num


def inverse_2q(num: Integral) -> Integral:
    """Return the inverse of a 2-qubit clifford integer."""
    return _CLIFFORD_INVERSE_2Q[num]


def num_from_2q_circuit(qc: QuantumCircuit) -> Integral:
    """Convert a given 2-qubit Clifford circuit to the corresponding integer.

    Note: The circuit must consist of gates in :const:`_CLIFF_SINGLE_GATE_MAP_2Q`,
    RZGate, Delay and Barrier.
    """
    lhs = 0
    for rhs in _clifford_2q_nums_from_2q_circuit(qc):
        lhs = _CLIFFORD_COMPOSE_2Q_DENSE[lhs, _clifford_num_to_dense_index[rhs]]
    return lhs


def _num_from_2q_gate(
    op: Instruction, qubits: Optional[Union[Tuple[int, int], Tuple[int]]] = None
) -> int:
    """
    Convert a given 1-qubit clifford operation to the corresponding integer.
    Note that supported operations are limited to ones in `_CLIFF_SINGLE_GATE_MAP_2Q` or Rz gate.

    Args:
        op: operation of instruction to be converted.
        qubits: qubits to which the operation applies

    Returns:
        An integer representing a Clifford consisting of a single operation.

    Raises:
        QiskitError: If the input instruction is not a Clifford instruction.
        QiskitError: If rz is given with a angle that is not Clifford.
    """
    if op.name in {"delay", "barrier"}:
        return 0

    qubits = qubits or (0, 1)
    try:
        name = _deparameterized_name(op)
        return _CLIFF_SINGLE_GATE_MAP_2Q[(name, qubits)]
    except QiskitError as err:
        raise QiskitError(
            f"Parameterized instruction {op.name} could not be converted to integer Clifford"
        ) from err
    except KeyError as err:
        raise QiskitError(
            f"Instruction {op.name} on {qubits} could not be converted to integer Clifford"
        ) from err


def _append_v_w(qc, vw0, vw1):
    if vw0 == "v":
        qc.sdg(0)
        qc.h(0)
    elif vw0 == "w":
        qc.h(0)
        qc.s(0)
    if vw1 == "v":
        qc.sdg(1)
        qc.h(1)
    elif vw1 == "w":
        qc.h(1)
        qc.s(1)


def _create_cliff_2q_layer_0():
    """Layer 0 consists of 0 or 1 H gates on each qubit, followed by 0/1/2 V gates on each qubit.
    Number of Cliffords == 36."""
    circuits = []
    num_h = [0, 1]
    v_w_gates = ["i", "v", "w"]
    for h0, h1, v0, v1 in itertools.product(num_h, num_h, v_w_gates, v_w_gates):
        qc = QuantumCircuit(2)
        for _ in range(h0):
            qc.h(0)
        for _ in range(h1):
            qc.h(1)
        _append_v_w(qc, v0, v1)
        circuits.append(qc)
    return circuits


def _create_cliff_2q_layer_1():
    """Layer 1 consists of one of the following:
    - nothing
    - cx(0,1) followed by 0/1/2 V gates on each qubit
    - cx(0,1), cx(1,0) followed by 0/1/2 V gates on each qubit
    - cx(0,1), cx(1,0), cx(0,1)
    Number of Cliffords == 20."""
    circuits = [QuantumCircuit(2)]  # identity at the beginning

    v_w_gates = ["i", "v", "w"]
    for v0, v1 in itertools.product(v_w_gates, v_w_gates):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        _append_v_w(qc, v0, v1)
        circuits.append(qc)

    for v0, v1 in itertools.product(v_w_gates, v_w_gates):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        _append_v_w(qc, v0, v1)
        circuits.append(qc)

    qc = QuantumCircuit(2)  # swap at the end
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    circuits.append(qc)
    return circuits


def _create_cliff_2q_layer_2():
    """Layer 2 consists of a Pauli gate on each qubit {Id, X, Y, Z}.
    Number of Cliffords == 16."""
    circuits = []
    pauli = ("i", XGate(), YGate(), ZGate())
    for p0, p1 in itertools.product(pauli, pauli):
        qc = QuantumCircuit(2)
        if p0 != "i":
            qc.append(p0, [0])
        if p1 != "i":
            qc.append(p1, [1])
        circuits.append(qc)
    return circuits


_CLIFFORD_LAYER = (
    _create_cliff_2q_layer_0(),
    _create_cliff_2q_layer_1(),
    _create_cliff_2q_layer_2(),
)
_NUM_LAYER_1 = 20
_NUM_LAYER_2 = 16


def _clifford_2q_nums_from_2q_circuit(qc: QuantumCircuit) -> Iterable[Integral]:
    """Yield Clifford numbers that represents the 2Q Clifford circuit."""
    for inst in qc:
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        yield _num_from_2q_gate(op=inst.operation, qubits=qubits)


# Construct mapping from Clifford layers to series of Clifford numbers
_CLIFFORD_LAYER_NUMS = [
    [tuple(_clifford_2q_nums_from_2q_circuit(qc)) for qc in _CLIFFORD_LAYER[layer]]
    for layer in [0, 1, 2]
]


@lru_cache(maxsize=256)
def _transformed_clifford_layer(
    layer: int,
    index: Integral,
    basis_gates: Tuple[str, ...],
    coupling_tuple: Optional[Tuple[Tuple[int, int]]],
    synthesis_method: str = DEFAULT_SYNTHESIS_METHOD,
) -> QuantumCircuit:
    # Return the index-th quantum circuit of the layer translated with the basis_gates.
    # The result is cached for speed.
    return _synthesize_clifford_circuit(
        _CLIFFORD_LAYER[layer][index],
        basis_gates=basis_gates,
        coupling_tuple=coupling_tuple,
        synthesis_method=synthesis_method,
    )


def _num_from_layer_indices(triplet: Tuple[Integral, Integral, Integral]) -> Integral:
    """Return the clifford number corresponding to the input triplet."""
    num = triplet[0] * _NUM_LAYER_1 * _NUM_LAYER_2 + triplet[1] * _NUM_LAYER_2 + triplet[2]
    return num


def _layer_indices_from_num(num: Integral) -> Tuple[Integral, Integral, Integral]:
    """Return the triplet of layer indices corresponding to the input number."""
    idx2 = num % _NUM_LAYER_2
    num = num // _NUM_LAYER_2
    idx1 = num % _NUM_LAYER_1
    idx0 = num // _NUM_LAYER_1
    return idx0, idx1, idx2


def _tensor_1q_nums(first: Integral, second: Integral) -> Integral:
    """Return the 2-qubit Clifford integer that is the tensor product of 1-qubit Cliffords."""
    return _CLIFFORD_TENSOR_1Q[first, second]
