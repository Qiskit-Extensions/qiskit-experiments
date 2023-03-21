# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Utilities for sampling layers in randomized benchmarking experiments
"""

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, List, Tuple, TypeVar
from numbers import Integral

import numpy as np
from numpy.random import Generator, default_rng, BitGenerator, SeedSequence
from functools import lru_cache

from qiskit import QuantumCircuit, QuantumRegister, Operator
from qiskit.exceptions import QiskitError
from qiskit.circuit import Gate
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import random_unitary, Pauli, random_pauli
from qiskit.extensions import UnitaryGate
from qiskit.converters import circuit_to_dag
from .clifford_utils import CliffordUtils, _clifford_1q_int_to_instruction
from .rb_experiment import SequenceElementType

from time import time

Idx = TypeVar("Idx", int, Tuple[int, ...])
Gate = TypeVar("Gate", Instruction, Operator, QuantumCircuit, BaseOperator)


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


class RBSampler(ABC):
    """Sampling distribution for randomized benchmarking experiments.
    Subclasses must implement the ``__call__()`` method."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, num_qubits, seed=None, **params) -> Tuple[Tuple[Idx, Gate], ...]:
        """Samplers should define this method such that it returns sampled layers
        given the input parameters. Each layer is represented by a list of size-2 tuples
        where the first element is a qubit index or a tuple of indices, and the second
        element represents the gate that should be applied."""
        pass


class SingleQubitSampler(RBSampler):
    def __call__(
        self,
        num_qubits,
        length,
        gate_set: Optional[Union[str, List]] = "clifford",
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    ) -> Tuple[Tuple[Idx, Gate], ...]:
        """Samples a layer of random single-qubit gates from a specified gate set.

        Returns:
        For two qubits and length 3, a possible output would be
        (((0,20),(1,14)),((0,11),(1,4)),((0,2),(1,9)))
        """
        rng = default_rng(seed=seed)
        if gate_set == "clifford":
            layers = []
            for i in rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT, size=(length, num_qubits)):
                layers.append(tuple(zip((range(num_qubits)), i)))
        elif gate_set == "pauli":
            layers = []
            for i in range(length):
                layers.append(tuple(zip(range(num_qubits), random_pauli(num_qubits, seed=rng))))
        else:
            # TODO finish this
            pass
        return layers


class EdgeGrabSampler(RBSampler):
    r"""The edge grab algorithm for sampling one- and two-qubit layers.

    # section: overview

        The edge grab sampler, given a list of :math:`w` qubits, their connectivity
        graph, and the desired two-qubit gate density :math:`\xi_s`, outputs a layer
        as follows:

            1. Begin with the empty set :math:`E` and :math:`E_r`, the set of all edges
               in the connectivity graph. Select an edge from :math:`E_r` at random and
               add it to :math:`E`, removing all edges that share a qubit with the edge
               from :math:`E_r`.
            2. Select edges from :math:`E` with the probability :math:`w\xi/2|E|`. These
               edges will have two-qubit gates in the output layer.

        This produces a layer with an expected two-qubit gate density :math:`\xi`. In
        the default mirror RB configuration where these layers are dressed with
        single-qubit Pauli layers, this means the overall two-qubit gate density will be
        :math:`\xi_s/2=\xi`. The overall density will converge to :math:`\xi` as the
        circuit size increases.

    # section: reference
        .. ref_arxiv:: 1 2008.11294

    """

    @timer_func
    def __call__(
        self,
        num_qubits: int,
        two_qubit_gate_density: float,
        coupling_map: List[List[int]],
        length: int,
        one_qubit_gate_set: Optional[Union[str, List]] = "clifford",
        two_qubit_gate_set: Optional[List] = ["cx"],
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    ) -> List[List]:
        """Sample layers using the edge grab algorithm.

        Args:
            num_qubits: The number of qubits to generate layers for.
            one_qubit_gate_set: The one qubit gate set to sample from. Can be either a
                list of gates or "clifford".
            two_qubit_gate_set: The two qubit gate set to sample from. Can be either a
                list of gates or one of "cx", "cy", "cz", or "csx".
            two_qubit_gate_density: the expected fraction of two-qubit gates in the
                sampled layer.
            coupling_map: List of pairs of connected edges between qubits.
            length: The length of the sequence to output.
            seed: Seed for random generation.

        Raises:
            Warning: If the coupling map has no connectivity or
                ``two_qubit_gate_density`` is too high.
            TypeError: If invalid gate set(s) are specified.

        Returns:
            A ``length``-long list of :class:`qiskit.circuit.QuantumCircuit` layers over
                ``num_qubits`` qubits. Each layer is represented by a list of tuples
                which are of two formats: (single qubit index, single qubit gate)
                represented by (int, int), or ((two qubit indices), two qubit gate)
                represented by ((int, int), int). Here's an example with the default
                choice of Cliffords for the single-qubit gates and CXs for the two-qubit
                gates:

                .. code-block::
                    (((1, 2), 0), (0, 12), (3, 20))

                This represents a layer where the 12th Clifford is performed on qubit 0,
                a CX is performed with control qubit 1 and target qubit 2, and the 20th
                Clifford is performed on qubit 3.

        """
        rng = default_rng(seed=seed)

        if isinstance(one_qubit_gate_set, list) or not (
            one_qubit_gate_set.casefold() == "clifford"
        ):
            raise TypeError("one_qubit_gate_set must be a list of gates or 'clifford'.")

        if num_qubits == 1:
            if one_qubit_gate_set.casefold() == "clifford":
                return (
                    ((0, i),) for i in rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT, size=length)
                )
            else:
                return enumerate((0, i) for i in rng.choice(one_qubit_gate_set, size=length))

        timed = {}

        timed["looptotal"] = time()
        timed["2q"] = 0
        timed["cliffs"] = 0

        qc_list = []
        for _ in list(range(length)):
            all_edges = coupling_map[:]  # make copy of coupling map from which we pop edges
            selected_edges = []
            while all_edges:
                rand_edge = all_edges.pop(rng.integers(len(all_edges)))
                selected_edges.append(
                    rand_edge
                )  # move random edge from all_edges to selected_edges
                old_all_edges = all_edges[:]
                all_edges = []
                # only keep edges in all_edges that do not share a vertex with rand_edge
                for edge in old_all_edges:
                    if rand_edge[0] not in edge and rand_edge[1] not in edge:
                        all_edges.append(edge)

            qr = QuantumRegister(num_qubits)
            qc = QuantumCircuit(qr)
            two_qubit_prob = 0
            try:
                # need to divide by 2 since each two-qubit gate spans two lattice sites
                two_qubit_prob = num_qubits * two_qubit_gate_density / 2 / len(selected_edges)
            except ZeroDivisionError:
                warnings.warn("Device has no connectivity. All gates will be single-qubit.")
            if two_qubit_prob > 1:
                warnings.warn(
                    "Mean number of two-qubit gates is higher than the number of selected edges. "
                    + "Actual density of two-qubit gates will likely be lower than input density."
                )

            # selected_edges_logical is selected_edges with logical qubit labels rather than physical
            # ones. Example: qubits = (8,4,5,3,7), selected_edges = [[4,8],[7,5]]
            # ==> selected_edges_logical = [[1,0],[4,2]]
            put_1_qubit_gates = np.arange(num_qubits)
            # put_1_qubit_gates is a list of qubits that aren't assigned to a 2-qubit gate
            # 1-qubit gates will be assigned to these edges
            t1 = time()

            for edge in selected_edges:
                if rng.random() < two_qubit_prob:
                    # with probability two_qubit_prob, place a two-qubit gate from the
                    # gate set on edge in selected_edges
                    if two_qubit_gate_set == ["cx"]:
                        qc.cx(edge[0], edge[1])
                    else:
                        try:
                            getattr(qc, rng.choice(two_qubit_gate_set))(edge[0], edge[1])
                        except AttributeError:
                            raise QiskitError("Invalid two-qubit gate set specified.")
                    # remove these qubits from put_1_qubit_gates
                    put_1_qubit_gates = np.setdiff1d(put_1_qubit_gates, edge)
            t2 = time()

            # for q in put_1_qubit_gates:
            #     if one_qubit_gate_set == "clifford":
            #         gates_1q = _clifford_1q_int_to_instruction
            #         gates_1q = CliffordUtils.clifford_1_qubit(
            #             rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT)
            #         ).to_instruction()
            #     else:
            #         gates_1q = rng.choice(one_qubit_gate_set).to_instruction()
            #     qc.compose(gates_1q, [q], inplace=True)

            for q in put_1_qubit_gates:
                gate_1q = rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT)

                qc.compose(_clifford_1q_int_to_instruction(gate_1q, basis_gates), [q], inplace=True)
            print(qc.decompose())
            qc_list.append(qc)
            t3 = time()
            timed["2q"] += t2 - t1
            timed["cliffs"] += t3 - t2
        timed["looptotal"] = time() - timed["looptotal"]
        print(timed)
        return qc_list
