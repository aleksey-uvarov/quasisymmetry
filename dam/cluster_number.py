import numpy as np
import ffsim
import scipy
from scipy.sparse.linalg import LinearOperator
from math import comb
from functools import cache

from utils import integers_to_phases_polynomial

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

    
def from_num_operator_to_expnum_operator(num_operator, max_num_eval):
    """Returns LinearOperator exp(i * pi * num_operator / (max_num_eval + 1)), built efficiently"""
    dim = num_operator.shape[0]
    zero_op = LinearOperator((dim, dim), matvec=lambda v: np.zeros_like(v))
    coeffs = integers_to_phases_polynomial(max_num_eval)
    summands = [coeffs[n] * (num_operator ** n) for n in range(max_num_eval+1)]
    return sum(summands, start=zero_op)

def number_matrix_to_operators(cluster_number_matrix: np.ndarray,
                                     norb,
                                     nelec,
                                     expnum=False):
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
        num_operator = sum(summands, start=zero_op)
        if expnum == False:
            operators.append(num_operator)
        else:
            max_num_eval = 2 * sum(cluster_number_matrix[i]) # max cluster number eval = 2 * number of orbitals in the cluster
            operators.append(from_num_operator_to_expnum_operator(num_operator, int(max_num_eval)))
    return(operators)

def number_and_parity_symmetry_sectors(cluster_number_matrix, cluster_parity_matrix, norb, nelec):
    # input shape checks
    if len(cluster_number_matrix) > 0:
        if cluster_number_matrix.shape[1] != norb:
            raise ValueError("cluster_number_matrix must be a list of length-norb binary lists")
        cluster_number_matrix_to_int = []
        for cluster in cluster_number_matrix:
            # convert clusters from binary to integers; can't do directly due to order
            cluster_int = 0
            for i, bit in enumerate(cluster):
                if bit:  # If the bit is 1
                    cluster_int |= (1 << i)  # Set the i-th bit to 1
            cluster_number_matrix_to_int.append(cluster_int)
    if len(cluster_parity_matrix) > 0:
        if cluster_parity_matrix.shape[1] != norb:
            raise ValueError("cluster_parity_matrix must be a list of length-norb binary lists")
        # same as above
        cluster_parity_matrix_to_int = []
        for cluster in cluster_parity_matrix:
            cluster_int = 0
            for i, bit in enumerate(cluster):
                if bit:  # If the bit is 1
                    cluster_int |= (1 << i)  # Set the i-th bit to 1
            cluster_parity_matrix_to_int.append(cluster_int)

    dim = comb(norb, nelec[0]) * comb(norb, nelec[1])  
    alpha_indices, beta_indices = ffsim.addresses_to_strings(
    range(dim), norb, nelec,
    bitstring_type=ffsim.BitstringType.INT,
    concatenate=False
    ) # integers corresponding to FLIPPED alpha/beta bitstrings of basis determinants
    sectors = {}  
    for i in range(dim):
        if len(cluster_number_matrix) == 0:
            sector_label_num = ()
        else:
            sector_label_num = (cluster_int & alpha_indices[i] + cluster_int & beta_indices[i] for cluster_int in cluster_number_matrix_to_int)
        if len(cluster_parity_matrix) == 0:
            sector_label_par = ()
        else:
            sector_label_num = ((cluster_int & alpha_indices[i] + cluster_int & beta_indices[i]) % 2 for cluster_int in cluster_parity_matrix_to_int)
        sector_label = (sector_label_num, sector_label_par)
        sectors.setdefault(sector_label, []).append(i)

    return sectors
    





    #TODO implement based on metrics.symmetry_sectors. Think it through. If len(cluster_number_matrix) == 0 or len(cluster_parity_matrix) == 0 (or both), you want to keep clean dict keys...

    return sectors

#TODO write code for the two scatterplots.