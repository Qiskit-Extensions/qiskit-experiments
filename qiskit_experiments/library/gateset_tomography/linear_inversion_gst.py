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
Gate Set Tomography Linear Inversion Fitter
"""

from typing import Dict
import numpy as np
from qiskit.quantum_info import PTM


def linear_inversion_gst(outcome_data, gateset_basis) -> Dict[str, PTM]:

    """
    Reconstruct a gate set from measurement data using linear inversion.

    Returns:
        For each gate in the gate set: its approximation found
        using the linear inversion process.

    Additional Information:
        Given a gate set (G1,...,Gm)
        and SPAM circuits (F1,...,Fn) constructed from those gates
        the data should contain the probabilities of the following types:
        p_ijk = meas*f_i*g_k*f_j*rho
        p_ij = meas*f_i*f_j*rho

        We have p_ijk = self.probs[(Fj, Gk, Fi)] since in self.probs
        (Fj, Gk, Fi) indicates first applying Fj, then Gk, then Fi.

        One constructs the Gram matrix g = (p_ij)_ij
        which can be described as a product g=AB
        where A = sum (i> <meas f_i) and B=sum (f_j rho><j)
        For each gate Gk one can also construct the matrix Mk=(pijk)_ij
        which can be described as Mk=A*Gk*B
        Inverting g we obtain g^-1 = B^-1A^-1 and so
        g^1 * Mk = B^-1 * Gk * B
        This gives us a matrix similiar to Gk's representing matrix.
        However, it will not be the same as Gk,
        since the observable results cannot distinguish
        between (G1,...,Gm) and (B^-1*G1*B,...,B^-1*Gm*B)
        a further step of *Gauge optimization* is required on the results
        of the linear inversion stage.
        One can also use the linear inversion results as a starting point
        for a MLE optimization for finding a physical gateset, since
        unless the probabilities are accurate, the resulting gateset
        need not be physical.
    """
    n = len(gateset_basis.spam_labels)
    m = len(gateset_basis.gate_labels)
    gram_matrix = np.zeros((n, n))
    meas = np.zeros((1, n))
    rho = np.zeros((n, 1))
    gate_matrices = []
    for i in range(m):
        gate_matrices.append(np.zeros((n, n)))

    for i in range(n):  # row
        f_i = gateset_basis.spam_labels[i]
        meas[0][i] = outcome_data[(f_i,)]
        rho[i][0] = outcome_data[(f_i,)]
        for j in range(n):  # column
            f_j = gateset_basis.spam_labels[j]
            gram_matrix[i][j] = outcome_data[(f_i, f_j)]

            for k in range(m):  # gate
                g_k = gateset_basis.gate_labels[k]
                gate_matrices[k][i][j] = outcome_data[(f_i, g_k, f_j)]

    gram_inverse = np.linalg.inv(gram_matrix)

    gates = [PTM(gram_inverse @ gate_matrix) for gate_matrix in gate_matrices]
    result = dict(zip(gateset_basis.gate_labels, gates))
    result["meas"] = meas
    result["rho"] = gram_inverse @ rho
    return result
