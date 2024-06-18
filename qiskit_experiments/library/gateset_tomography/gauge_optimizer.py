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
Gauge optimizer for linear inversion GST results
"""

import itertools
from typing import Dict
import numpy as np
import scipy.optimize as opt
from qiskit.quantum_info import PTM, Choi, Pauli
from .gatesetbasis import GateSetBasis


def gauge_optimizer(
    initial_gateset: Dict, gateset_basis: GateSetBasis, num_qubits: int
) -> Dict[str, PTM]:
    """Initialize gauge optimizer fitter with the ideal and expected
        outcomes.
    Args:
        initial_gateset: The experimentally-obtained gate approximations using linear
        inversion method.
        gateset_basis: The gateset data
        num_qubits: number of qubits GST experiment is performed on.

    Additional information:
        Gauge optimization aims to find a basis in which the tomography
        results are as close as possible to the ideal (noiseless) results.

        Given a gateset specification (E, rho, G1,...,Gn) and any
        invertible matrix B, the gateset specification
        (E*B^-1, B*rho, B*G1*B^-1,...,B*Gn*B^-1)
        is indistinguishable from it by the tomography results.

        B is called the gauge matrix and the goal of gauge optimization
        is finding the B for which the resulting gateset description
        is optimal in some sense; we choose to minimize the norm
        difference between the gates found by experiment
        and the "expected" gates in the ideal (noiseless) case.
    Returns:
        The gate set after the gauge optimization.
    """

    # Obtaining the ideal gateset
    ideal_gateset = ideal_gateset_gen(gateset_basis, num_qubits, "PTM")
    fs = [gateset_basis.spam_matrix(label) for label in gateset_basis.spam_labels]
    d = np.shape(ideal_gateset["rho"])[0]
    rho = ideal_gateset["rho"]
    initial_value = np.array([(F @ rho).T[0] for F in fs]).T
    result = opt.minimize(
        obj_fn,
        x0=np.real(initial_value),
        args={
            "d": d,
            "initial_gateset": initial_gateset,
            "gateset_basis": gateset_basis,
            "ideal_gateset": ideal_gateset,
        },
    )
    return x_to_gateset(result.x, d, initial_gateset, gateset_basis)


def x_to_gateset(
    x: np.array, d: int, initial_gateset: Dict[str, PTM], gateset_basis: GateSetBasis
) -> Dict[str, PTM]:
    """Converts the gauge to the gateset defined by it
    Args:
        x: An array representation of the b matrix
        d: The Hilbert-space dimension, which is equal to 2 to the power of the number of qubits.
        initial_gateset: Experimental results for the gate set data obtained by linear inversion.
        gateset_basis: The gateset data

    Returns:
        The gateset obtained from b

    Additional information:
        Given a vector representation of b, this functions
        produces the list [b*G1*b^-1,...,b*Gn*b^-1]
        of gates correpsonding to the gauge b

    """
    b = np.array(x).reshape((d, d))  # b is the gauge (B)
    try:
        bb = np.linalg.inv(b)
    except np.linalg.LinAlgError:
        return None
    gateset = {
        label: PTM(b @ np.real(initial_gateset[label].data) @ bb)
        for label in gateset_basis.gate_labels
    }
    gateset[("meas")] = np.real(initial_gateset["meas"]) @ bb
    gateset["rho"] = b @ np.real(initial_gateset["rho"])
    return gateset


def obj_fn(x: np.array, args: Dict) -> float:
    """The norm-based score function for the gauge optimizer
    Args:
        x: An array representation of the B matrix

        args: A dict of the needed arguments that define the objective function and it includes:
        'd'- the Hilbert space dimension, 'initial_gateset'- the gateset obtained by linear inversion,
        'gateset_basis'- the gateset data and 'ideal_gateset'- the noiseless gateset.

    Returns:
        The sum of norm differences between the ideal gateset
        and the one corresponding to B
    """
    d, initial_gateset, gateset_basis, ideal_gateset = (
        args["d"],
        args["initial_gateset"],
        args["gateset_basis"],
        args["ideal_gateset"],
    )
    gateset = x_to_gateset(x, d, initial_gateset, gateset_basis)
    result = sum(
        [
            np.linalg.norm(np.real(gateset[label].data) - np.real(ideal_gateset[label].data))
            for label in gateset_basis.gate_labels
        ]
    )
    result = result + np.linalg.norm(np.real(gateset["meas"]) - np.real(ideal_gateset["meas"]))
    result = result + np.linalg.norm(np.real(gateset["rho"]) - np.real(ideal_gateset["rho"]))
    return result


def default_init_state(num_qubits):
    """Returns the PTM representation of the usual ground state |00...>"""
    d = np.power(2, num_qubits)

    # matrix representation of #rho in regular Hilbert space
    matrix_init_0 = np.zeros((d, d), dtype=complex)
    matrix_init_0[0, 0] = 1

    # decompoition into Pauli strings basis (PTM representation)
    matrix_init_pauli = [
        np.trace(np.dot(matrix_init_0, pauli_strings(num_qubits)[i])) for i in range(np.power(d, 2))
    ]
    return np.reshape(matrix_init_pauli, (np.power(d, 2), 1))


def default_measurement_op(num_qubits):
    """The PTM representation of the usual Z-basis measurement"""
    d = np.power(2, num_qubits)

    # matrix representation of #E=|00..><00...| in regular Hilbert space
    matrix_meas_0 = np.zeros((d, d), dtype=complex)
    matrix_meas_0[0, 0] = 1

    # decompoition into Pauli strings basis (PTM representation)
    matrix_meas_pauli = [
        np.trace(np.dot(matrix_meas_0, pauli_strings(num_qubits)[i])) for i in range(np.power(d, 2))
    ]
    return matrix_meas_pauli


def ideal_gateset_gen(gateset_basis, num_qubits, type_pt_ch):
    """Generate the ideal gate set"""
    # type takes a string either 'PTM' or 'Choi'
    # generates the ideal (noiseless) gate set.
    ideal_gateset_ptm = {
        label: PTM(gateset_basis.gate_matrices[label]) for label in gateset_basis.gate_labels
    }
    ideal_gateset_choi = {
        label: Choi(PTM(gateset_basis.gate_matrices[label])) for label in gateset_basis.gate_labels
    }
    ideal_gateset_ptm["meas"] = default_measurement_op(num_qubits)
    ideal_gateset_ptm["rho"] = default_init_state(num_qubits)
    ideal_gateset_choi["meas"] = default_measurement_op(num_qubits)
    ideal_gateset_choi["rho"] = default_init_state(num_qubits)
    return ideal_gateset_ptm if type_pt_ch == "PTM" else ideal_gateset_choi


def pauli_strings(num_qubits):
    """Return the normalized matrix representation of Pauli strings basis
    of size=num_qubits. e.g., for num_qubits=2, it returns the matrix representations
    of 0.5*['II','IX','IY','IZ,'XI','YI',...]"""

    pauli_labels = ["I", "X", "Y", "Z"]
    pauli_strings_matrices = [
        Pauli("".join(p)).to_matrix() for p in itertools.product(pauli_labels, repeat=num_qubits)
    ]
    # normalization
    pauli_strings_normalized = [
        (1 / np.sqrt(2 ** num_qubits)) * pauli_strings_matrices[i]
        for i in range(len(pauli_strings_matrices))
    ]
    return pauli_strings_normalized
