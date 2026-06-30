import numpy as np
import pytest
import scipy
import ffsim
from cluster_number import build_one_orb_num_operators, build_two_orb_num_operators, cluster_matrix_to_cluster_number_quasisymmetries

def test_spin_orbital_occupations():
    """Testing to get familiar with scipy/ffsim basis ordering and operator usage"""
    norb = 3
    nelec = (1, 2) # alpha, beta
    # generate spin-orbital number operators
    alpha_number_operators = []
    beta_number_operators = []
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
        alpha_number_operators.append(ffsim.linear_operator(n_alpha, norb, nelec))
        beta_number_operators.append(ffsim.linear_operator(n_beta, norb, nelec))

    spin_orb_number_operators = alpha_number_operators + beta_number_operators

    # expected basis ordering: |alpha occupations -> 100; beta occupations -> 110>, |100; 101>, ...
    expected_occupations = [[1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 1], 
                            [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1], 
                            [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1]]

    # get actual basis ordering
    for i in range(9):
        basis_elem = np.eye(9)[i]
        occupations = [np.vdot(basis_elem, op @ basis_elem) for op in spin_orb_number_operators]
        assert list(occupations) == expected_occupations[i]

def test_build_oneortwo_orb_num_operators():
    norb = 3
    nelec = (1, 2) # alpha, beta
    # get list of one-orbital alpha+beta occ number operators
    one_orbital_num_operators = build_one_orb_num_operators(norb, nelec)

    expected_occupations = [[2, 1, 0], [2, 0, 1], [1, 1, 1], 
                            [1, 2, 0], [1, 1, 1], [0, 2, 1], 
                            [1, 1, 1], [1, 0, 2], [0, 1, 2]]
    
    assert len(one_orbital_num_operators) == norb

    assert type(one_orbital_num_operators[0]) == scipy.sparse.linalg._interface._CustomLinearOperator

    assert np.allclose(one_orbital_num_operators[0] @ np.eye(9)[0] - 2 * np.eye(9)[0], np.zeros(9))

    two_orbital_num_operators = build_two_orb_num_operators(norb, nelec) # orbital pairs: 01, 02, 12

    assert len(two_orbital_num_operators) == (norb**2 - norb)/2

    expected_two_orb_occupations = [[3, 2, 1], [2, 3, 1], [2, 2, 2], 
                                [3, 1, 2], [2, 2, 2], [2, 1, 3], 
                                [2, 2, 2], [1, 3, 2], [1, 2, 3]]

    for i in range(9):
        basis_elem = np.eye(9)[i]
        occupations = [np.vdot(basis_elem, op @ basis_elem) for op in one_orbital_num_operators]
        two_orb_occupations = [np.vdot(basis_elem, op @ basis_elem) for op in two_orbital_num_operators]
        assert list(occupations) == expected_occupations[i]
        assert list(two_orb_occupations) == expected_two_orb_occupations[i]


def test_cluster_matrix_to_cluster_number_quasisymmetries():
    norb = 3
    nelec = (1, 2)
    cluster_matrix = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 1]])
    cluster_num_operators = cluster_matrix_to_cluster_number_quasisymmetries(cluster_matrix, norb, nelec)
    expected_cluster_num_operators = [build_one_orb_num_operators(norb, nelec)[0], build_two_orb_num_operators(norb, nelec)[1], 3 * np.eye(9)]
    for i in range(9):
        basis_elem = np.eye(9)[i]
        for j in range(3):
            assert np.allclose(cluster_num_operators[j] @ basis_elem, expected_cluster_num_operators[j] @ basis_elem)



# def test_cluster_number_symmetry_sectors: #TODO
