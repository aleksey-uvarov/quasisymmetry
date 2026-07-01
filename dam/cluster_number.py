import argparse
import numpy as np
import time
import ffsim
import scipy
import pyscf
import pyscf.fci
from scipy.sparse.linalg import LinearOperator

from typing import Callable
from math import comb
from functools import cache, reduce

# from chemistry import load_moldata, fcidump_data # may need to modify path

@cache # consider modifying output to tuple, so cached output will be immutable
def build_one_orb_num_operators(norb, nelec):
    """Returns list of single-orbital occupation number operators (alpha + beta)"""
    orb_number_operators = []
    for i in range(norb):
        n_alpha = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(i)): +1 # 1 * a_ialpha^dag a_ialpha
            }
        )
        n_beta = ffsim.FermionOperator(
            {
                (ffsim.cre_b(i), ffsim.des_b(i)): +1 # 1 * a_ibeta^dag a_ibeta
            }
        )
        n = n_alpha + n_beta
        orb_number_operators.append(ffsim.linear_operator(n, norb, nelec))
    return orb_number_operators

def build_two_orb_num_operators(norb, nelec):
    """Returns list of two-orbital occupation number operators (alpha + beta)."""
    one_orb_num_operators = build_one_orb_num_operators(norb, nelec)
    two_orb_number_operators = []
    for i in range(norb):
        for j in range(i+1, norb):
            two_orb_number_operators.append(one_orb_num_operators[i] + one_orb_num_operators[j])
    return two_orb_number_operators


def number_matrix_to_operators(cluster_number_matrix: np.ndarray,
                                     norb,
                                     nelec):
    """Returns a list of cluster number operators. The orbitals of the i th operator correspond to the 1's in the ith row of the binary cluster_number_matrix."""
    if len(cluster_number_matrix) == 0:
        return([])
    # Probably this won't be a bottleneck, but may want to avoid multiple 
    # one_orb_num_operators calls (see build_two_orb_num_operators)
    if cluster_number_matrix.shape[1] != norb:
        raise ValueError("shape[1] of cluster_number_matrix must equal norb")
    
    # get one-orbital number operators
    one_orb_num_operators = build_one_orb_num_operators(norb, nelec)

    # add those to cluster number operators
    operators = [] # will contain the cluster number/quasisymmetry operators
    dim = scipy.special.comb(norb, nelec[0], exact=True) * scipy.special.comb(norb, nelec[1], exact=True)  # Hilbert space dim
    zero_op = LinearOperator((dim, dim), matvec=lambda v: np.zeros_like(v))
    for i in range(cluster_number_matrix.shape[0]):
        summands = [one_orb_num_operators[j] for j in range(norb) if cluster_number_matrix[i][j] == 1]
        operators.append(sum(summands, start=zero_op))
    return(operators)

def x_to_rotation(x, norb):
    iu = np.triu_indices(norb, k=1)
    rotation_generator = np.zeros((norb, norb))
    rotation_generator[iu] = x
    rotation_generator -= rotation_generator.T
    return scipy.linalg.expm(rotation_generator)

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
            total_nc += np.linalg.norm(commutator_on_state)**2
        return total_nc
    return f

#TODO implement based on metrics.symmetry_sectors
# def cluster_number_symmetry_sectors(cluster_number_matrix, norb, nelec):
    

#TODO write code for the two scatterplots.