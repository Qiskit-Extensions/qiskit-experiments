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
Constrained convex least-squares tomography fitter.
"""

from typing import Optional, Dict, Tuple, Union
import numpy as np

from qiskit_experiments.library.tomography.basis import (
    MeasurementBasis,
    PreparationBasis,
)
from . import cvxpy_utils
from .cvxpy_utils import cvxpy
from . import lstsq_utils


@cvxpy_utils.requires_cvxpy
def cvxpy_linear_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int, ...]] = None,
    preparation_qubits: Optional[Tuple[int, ...]] = None,
    conditional_measurement_indices: Optional[Tuple[int, ...]] = None,
    psd: bool = True,
    trace_preserving: bool = False,
    trace: Optional[float] = None,
    partial_trace: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    r"""Constrained weighted linear least-squares tomography fitter.

    Overview
        This fitter reconstructs the maximum-likelihood estimate by using
        ``cvxpy`` to minimize the constrained least-squares negative log
        likelihood function

        .. math::
            \hat{\rho}
                &= -\mbox{argmin }\log\mathcal{L}{\rho} \\
                &= \mbox{argmin }\sum_i w_i^2(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2 \\
                &= \mbox{argmin }\|W(Ax - y) \|_2^2

        subject to

        - *Positive-semidefinite* (``psd=True``): :math:`\rho \gg 0` is constrained
          to be a positive-semidefinite matrix.
        - *Trace* (``trace=t``): :math:`\mbox{Tr}(\rho) = t` is constrained to have
          the specified trace.
        - *Trace preserving* (``trace_preserving=True``): When performing process
          tomography the Choi-state :math:`\rho` represents is constrained to be
          trace preserving.

        where

        - :math:`A` is the matrix of measurement operators
          :math:`A = \sum_i |i\rangle\!\langle\!\langle M_i|`
        - :math:`y` is the vector of expectation value data for each projector
          corresponding to estimates of :math:`b_i = Tr[M_i \cdot x]`.
        - :math:`x` is the vectorized density matrix (or Choi-matrix) to be fitted
          :math:`x = |\rho\rangle\\!\rangle`.

    .. note:

        Various solvers can be called in CVXPY using the `solver` keyword
        argument. When ``psd=True`` the optimization problem is a case of a
        *semidefinite program* (SDP) and requires a SDP compatible solver
        for CVXPY. CVXPY includes an SDP compatible solver `SCS`` but it
        is recommended to install the the open-source ``CVXOPT`` solver
        or one of the supported commercial solvers. See the `CVXPY
        documentation
        <https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options>`_
        for more information on solvers.

    .. note::

        Linear least-squares constructs the full basis matrix :math:`A` as a dense
        numpy array so should not be used for than 5 or 6 qubits. For larger number
        of qubits try the
        :func:`~qiskit_experiments.library.tomography.fitters.linear_inversion`
        fitter function.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: Optional, measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
        measurement_qubits: Optional, the physical qubits that were measured.
            If None they are assumed to be ``[0, ..., M-1]`` for M measured qubits.
        preparation_qubits: Optional, the physical qubits that were prepared.
            If None they are assumed to be ``[0, ..., N-1]`` for N preparated qubits.
        conditional_measurement_indices: Optional, conditional measurement data
            indices. If set this will return a list of conditional fitted states
            conditioned on a fixed basis measurement of these qubits.
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be trace preserving when
                          fitting a Choi-matrix in quantum process
                          tomography (default: False).
        partial_trace: Enforce conditional fitted Choi matrices to partial
                       trace to POVM matrices.
        trace: trace constraint for the fitted matrix (default: None).
        weights: Optional array of weights for least squares objective.
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPy is not installed on the current system.
        AnalysisError: If analysis fails.

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    if measurement_basis and measurement_qubits is None:
        measurement_qubits = tuple(range(measurement_data.shape[1]))
    if preparation_basis and preparation_qubits is None:
        preparation_qubits = tuple(range(preparation_data.shape[1]))

    basis_matrix, probability_data = lstsq_utils.lstsq_data(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        measurement_qubits=measurement_qubits,
        preparation_qubits=preparation_qubits,
        conditional_measurement_indices=conditional_measurement_indices,
    )

    # Since CVXPY only works with real variables we must specify the real
    # and imaginary parts of matrices seperately: rho = rho_r + 1j * rho_i

    num_circ_components, num_tomo_components, _ = probability_data.shape
    dim = int(np.sqrt(basis_matrix.shape[1]))

    # Generate list of conditional components for block diagonal matrix
    # rho = sum_k |k><k| \otimes rho(k)
    rhos_r = []
    rhos_i = []
    cons = []
    metadata = {"component_conditionals": []}
    for i in range(num_circ_components):
        for j in range(num_tomo_components):
            rho_r, rho_i, cons_i = cvxpy_utils.complex_matrix_variable(dim, hermitian=True, psd=psd)
            rhos_r.append(rho_r)
            rhos_i.append(rho_i)
            cons.append(cons_i)
            metadata["component_conditionals"].append((i, j))

    # Partial trace when fitting Choi-matrices for quantum process tomography.
    # This applied to the sum of conditional components
    # Note that this adds an implicitly
    # trace preserving is a specific partial trace constraint ptrace(rho) = I
    # Note: partial trace constraints implicitly define a trace constraint,
    # so if a different trace constraint is specified it will be ignored
    joint_cons = None
    if partial_trace is not None:
        for rho_r, rho_i, povm in zip(rhos_r, rhos_i, partial_trace):
            joint_cons = cvxpy_utils.partial_trace_constaint(rho_r, rho_i, povm)
    elif trace_preserving:
        if not preparation_qubits:
            preparation_qubits = tuple(range(preparation_data.shape[1]))
        input_dim = np.prod(preparation_basis.matrix_shape(preparation_qubits))
        joint_cons = cvxpy_utils.trace_preserving_constaint(
            rhos_r,
            rhos_i,
            input_dim=input_dim,
            hermitian=True,
        )
    elif trace is not None:
        joint_cons = cvxpy_utils.trace_constraint(rhos_r, rhos_i, trace=trace, hermitian=True)

    # OBJECTIVE FUNCTION

    # The function we wish to minimize is || arg ||_2 where
    #   arg =  bm * vec(rho) - data
    # Since we are working with real matrices in CVXPY we expand this as
    #   bm * vec(rho) = (bm_r + 1j * bm_i) * vec(rho_r + 1j * rho_i)
    #                 = bm_r * vec(rho_r) - bm_i * vec(rho_i)
    #                   + 1j * (bm_r * vec(rho_i) + bm_i * vec(rho_r))
    #                 = bm_r * vec(rho_r) - bm_i * vec(rho_i)
    # where we drop the imaginary part since the expectation value is real

    # Construct block diagonal fit variable from conditional components
    # Construct objective function
    if weights is not None:
        weights = weights / np.sqrt(np.sum(weights**2))
        probability_data = weights * probability_data

    if weights is None:
        bm_r = np.real(basis_matrix)
        bm_i = np.imag(basis_matrix)
        bms_r = [bm_r] * num_circ_components * num_tomo_components
        bms_i = [bm_i] * num_circ_components * num_tomo_components
    else:
        bms_r = []
        bms_i = []
        for i in range(num_circ_components):
            for j in range(num_tomo_components):
                weighted_mat = weights[i, j][:, None] * basis_matrix
                bms_r.append(np.real(weighted_mat))
                bms_i.append(np.imag(weighted_mat))

    # Stack lstsq objective from sum of components
    args = []
    idx = 0
    for i in range(num_circ_components):
        for j in range(num_tomo_components):
            model = bms_r[idx] @ cvxpy.vec(rhos_r[idx]) - bms_i[idx] @ cvxpy.vec(rhos_i[idx])
            data = probability_data[i, j]
            args.append(model - data)
            idx += 1

    # Combine all variables and constraints into a joint optimization problem
    # if tehre is a joint constraint
    if joint_cons:
        args = [cvxpy.hstack(args)]
        for cons_i in cons:
            joint_cons += cons_i
        cons = [joint_cons]

    # Solve each component separately
    metadata = {
        "cvxpy_solver": None,
        "cvxpy_status": [],
    }
    for arg, con in zip(args, cons):
        # Optimization problem
        obj = cvxpy.Minimize(cvxpy.norm(arg, p=2))
        prob = cvxpy.Problem(obj, con)

        # Solve SDP
        cvxpy_utils.solve_iteratively(prob, 5000, **kwargs)

        # Return optimal values and problem metadata
        metadata["cvxpy_solver"] = prob.solver_stats.solver_name
        metadata["cvxpy_status"].append(prob.status)

    if psd:
        metadata["psd_constraint"] = True
    if partial_trace is not None:
        metadata["ptr_constraint"] = partial_trace
    elif trace_preserving:
        metadata["tp_constraint"] = True
    elif trace is not None:
        metadata["trace_constraint"] = trace

    fits = [rho_r.value + 1j * rho_i.value for rho_r, rho_i in zip(rhos_r, rhos_i)]
    return fits, metadata


@cvxpy_utils.requires_cvxpy
def cvxpy_gaussian_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int, ...]] = None,
    preparation_qubits: Optional[Tuple[int, ...]] = None,
    conditional_measurement_indices: Optional[Tuple[int, ...]] = None,
    psd: bool = True,
    trace_preserving: bool = False,
    trace: Optional[float] = None,
    outcome_prior: Union[np.ndarray, int] = 0.5,
    **kwargs,
) -> Dict:
    r"""Constrained Gaussian linear least-squares tomography fitter.

    .. note::

        This function calls :func:`cvxpy_linear_lstsq` with a Gaussian weights
        vector. Refer to its documentation for additional details.

    Overview
        This fitter reconstructs the maximum-likelihood estimate by using
        ``cvxpy`` to minimize the constrained least-squares negative log
        likelihood function

        .. math::
            \hat{\rho}
                &= \mbox{argmin} (-\log\mathcal{L}{\rho}) \\
                &= \mbox{argmin }\|W(Ax - y) \|_2^2 \\
            -\log\mathcal{L}(\rho)
                &= |W(Ax -y) \|_2^2 \\
                &= \sum_i \frac{1}{\sigma_i^2}(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2

    Additional Details
        The Gaussian weights are estimated from the observed frequency and shot data
        via a Bayesian update of a Dirichlet distribution with observed outcome data
        frequences :math:`f_i(s)`, and Dirichlet prior :math:`\alpha_i(s)` for
        tomography basis index `i` and measurement outcome `s`.

        The mean posterior probabilities are computed as

        .. math:
            p_i(s) &= \frac{f_i(s) + \alpha_i(s)}{\bar{\alpha}_i + N_i} \\
            Var[p_i(s)] &= \frac{p_i(s)(1-p_i(s))}{\bar{\alpha}_i + N_i + 1}
            w_i(s) = \sqrt{Var[p_i(s)]}^{-1}

        where :math:`N_i = \sum_s f_i(s)` is the total number of shots, and
        :math:`\bar{\alpha}_i = \sum_s \alpha_i(s)` is the norm of the prior.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: Optional, measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
        measurement_qubits: Optional, the physical qubits that were measured.
            If None they are assumed to be ``[0, ..., M-1]`` for M measured qubits.
        preparation_qubits: Optional, the physical qubits that were prepared.
            If None they are assumed to be ``[0, ..., N-1]`` for N preparated qubits.
        conditional_measurement_indices: Optional, conditional measurement data
            indices. If set this will return a list of conditional fitted states
            conditioned on a fixed basis measurement of these qubits.
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be
            trace preserving when fitting a Choi-matrix in quantum process
            tomography (default: False).
        trace: trace constraint for the fitted matrix (default: None).
        outcome_prior: The Baysian prior :math:`\alpha` to use computing Gaussian
            weights. See additional information.
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPY is not installed on the current system.
        AnalysisError: If analysis fails.

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    _, variance = lstsq_utils.dirichlet_mean_and_var(
        outcome_data,
        shot_data=shot_data,
        outcome_prior=outcome_prior,
        conditional_measurement_indices=conditional_measurement_indices,
    )
    weights = 1.0 / np.sqrt(variance)
    return cvxpy_linear_lstsq(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        measurement_qubits=measurement_qubits,
        preparation_qubits=preparation_qubits,
        conditional_measurement_indices=conditional_measurement_indices,
        psd=psd,
        trace=trace,
        trace_preserving=trace_preserving,
        weights=weights,
        **kwargs,
    )
