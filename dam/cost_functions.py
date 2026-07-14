import numpy as np
import ffsim
import scipy
from scipy.sparse.linalg import LinearOperator
from typing import Callable

def x_to_rotation(x, norb):
    iu = np.triu_indices(norb, k=1)
    rotation_generator = np.zeros((norb, norb))
    rotation_generator[iu] = x
    rotation_generator -= rotation_generator.T
    return scipy.linalg.expm(rotation_generator)

# this version of commutator_cost is slightly more efficient as h_on_rotate_state
# is computed only once
def commutator_cost_v2(moldata: ffsim.MolecularData,
                    symmetries: list,
                    reference_state: np.ndarray) -> Callable:
    def f(x):
        U = x_to_rotation(x, moldata.norb)
        rotated_state = ffsim.apply_orbital_rotation(reference_state,
                                                     U,
                                                     moldata.norb,
                                                     moldata.nelec)
        h = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                  norb=moldata.norb, nelec=moldata.nelec)
        total_nc = 0
        h_on_rotate_state = h @ rotated_state
        for s in symmetries:
            term1 = h @ (s @ rotated_state)
            term2 = s @ h_on_rotate_state
            commutator_on_state = term1 - term2
            total_nc += np.vdot(commutator_on_state, commutator_on_state).real
        return total_nc
    return f

# this version of variance_cost works for arbitrary symmetry ops, not only parity ops (i.e., spec = {-1, 1}).
# Set only_parities=True for efficient parity-only version
def variance_cost_general(moldata: ffsim.MolecularData,
                    symmetries: list[scipy.sparse.linalg.LinearOperator],
                    reference_state: np.ndarray, only_parities=False) -> Callable:
    def f(x: np.ndarray) -> float:
        U = x_to_rotation(x, moldata.norb)
        rotated_state = ffsim.apply_orbital_rotation(reference_state,
                                                     U,
                                                     moldata.norb,
                                                     moldata.nelec)
        total_var = 0
        for s in symmetries:
            if only_parities: # use simplified expression
                total_var += 1 - ((rotated_state.T.conj() @ s @ rotated_state)**2).real
            else:
                total_var += (reference_state.T.conj() @ (s @ (s @ reference_state)) - (reference_state.T.conj() @ (s @ reference_state)) ** 2).real
        return total_var
    return f

# eigenvalue equation-based cost function
# Set only_parities=True for efficient only_parities-only version
def eval_eq_cost(symmetries: list, evals: list,
                    reference_state: np.ndarray, norb:int, nelec:int, only_parities=False) -> Callable:
    if len(symmetries) != len(evals):
        raise ValueError("len(evals) must match len(symmetries)")
    def f(x):
        U = x_to_rotation(x, norb)
        rotated_state = ffsim.apply_orbital_rotation(reference_state,
                                                     U,
                                                     norb,
                                                     nelec)
        total = 0
        for i in range(len(symmetries)):
            if only_parities: # use simplified expression
                total += 2 * (1 - evals[i] * np.vdot(rotated_state, symmetries[i] @ rotated_state).real)
            else:
                vec = symmetries[i] @ rotated_state - evals[i] * rotated_state
                total += np.vdot(vec, vec).real 
        return total
    return f

# compare with show_symmetries.py, lines 44-62
def get_heatmap_data_nc_score(h, ref_state, norb, diag_operators, off_diag_operators, upscale_factor=1):
    """Returns matrix of nc_scores, with entries ||[H, quasisymmetry_operator]|Ψ>||²"""
    # Initialize heatmap data
    nc_scores = np.zeros((norb, norb))

    # precompute H|psi>
    h_on_ref_state = h @ ref_state

    # Diagonal: Possibility to scale up with upscale_factor, for better visualization
    for i in range(norb):
        term1 = h @ (diag_operators[i] @ ref_state)
        term2 = diag_operators[i] @ h_on_ref_state # using the wrapper variable type. Commutator has same/related type
        commutator_on_state = term1 - term2
        nc_scores[i, i] = upscale_factor * np.linalg.norm(commutator_on_state)**2

    # Off-diagonal
    # Need to arefully handle indices (e.g., do inverse of index mapping of what is done in cluster_number.build_two_orb_num_operators)
    for i in range(norb):
        for j in range(i+1, norb):
            flat_index = i * norb - i * (i + 1) // 2 + j - i - 1 # checked
            op_ij = off_diag_operators[flat_index]
            term1 = h @ (op_ij @ ref_state)
            term2 = op_ij @ h_on_ref_state
            commutator_on_state = term1 - term2
            nc_scores[i, j] = np.linalg.norm(commutator_on_state)**2

    return nc_scores

def get_heatmap_data_variance_score(ref_state, norb, diag_operators, off_diag_operators, upscale_factor=1):
    variance_scores = np.zeros((norb, norb))

    for i in range(norb):
        variance_scores[i, i] = upscale_factor * ( (ref_state.T.conj() @ (diag_operators[i] @ (diag_operators[i] @ ref_state))) - ((ref_state.T.conj() @ (diag_operators[i] @ ref_state)) ** 2).real)

    for i in range(norb):
        for j in range(i + 1, norb):
            flat_index = i * norb - i * (i + 1) // 2 + j - i - 1
            op_ij = off_diag_operators[flat_index]
            variance_scores[i, j] = (ref_state.T.conj() @ (op_ij @ (op_ij @ ref_state)) - (ref_state.T.conj() @ (op_ij @ ref_state)) ** 2).real

    return variance_scores

def get_heatmap_data_eval_eq_score(ref_state, norb, diag_operators, off_diag_operators, diag_evals, off_diag_evals, upscale_factor=1):
    """Returns matrix of eigenvalue equation-based scores, with entries ||(quasisymmetry operator - eval * I)|Ψ>||²"""
    # Initialize heatmap data
    eval_scores = np.zeros((norb, norb))

    # Diagonal: Possibility to scale up with upscale_factor, for better visualization
    for i in range(norb):
        eval_scores[i, i] = upscale_factor * np.linalg.norm(diag_operators[i] @ ref_state - diag_evals[i] * ref_state)**2

    # Off-diagonal
    # Need to arefully handle indices (e.g., do inverse of index mapping of what is done in cluster_number.build_two_orb_num_operators)
    for i in range(norb):
        for j in range(i+1, norb):
            flat_index = i * norb - i * (i + 1) // 2 + j - i - 1 # checked
            op_ij = off_diag_operators[flat_index]
            off_diag_eval = off_diag_evals[flat_index]
            eval_scores[i, j] = np.linalg.norm(op_ij @ ref_state - off_diag_eval * ref_state)**2

    return eval_scores