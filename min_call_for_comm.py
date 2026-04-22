import sys
sys.path.insert(0, "..")
# from scipy.optimize import minimize, OptimizeResult
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.sparse.linalg as spla
import time


from optimize_for_commutator import get_mol, commutator_cost_function_fixed_abc
from optimization_different_abc import build_U_from_thetas, popcount
import numpy as np

import openfermion as of

def commutator_cost(H, psi):
    n_spin_orbitals = int(np.log2(psi.shape[0]))
    n_orbitals = n_spin_orbitals // 2
    pairs = list(combinations(range(n_orbitals), 2))
    m = len(pairs)

    def f(x):
        a = np.sin(x[m]) * np.cos(x[m + 1])
        b = np.sin(x[m]) * np.sin(x[m + 1])
        c = np.cos(x[m])
        U = build_U_from_thetas(n_orbitals, x[:m], pairs)
        symmetries = [(of.FermionOperator(((2 * i, 1), (2 * i, 0)), a)
                       + of.FermionOperator(((2 * i + 1, 1), (2 * i + 1, 0)), b)
                       + (of.FermionOperator(((2 * i, 1), (2 * i, 0)), c)
                       * of.FermionOperator(((2 * i + 1, 1), (2 * i + 1, 0)), 1)))
                      for i in range(n_orbitals)]

        symmetry_inter_ops = [of.get_interaction_operator(s, n_qubits=n_spin_orbitals)
                              for s in symmetries]

        total = 0
        for op in symmetry_inter_ops:
            op.rotate_basis(U.T.conj())
            op = of.get_fermion_operator(op)
            op_mat = of.get_sparse_operator(op, n_qubits=n_spin_orbitals)
            # print(op_mat.shape)
            comm = H @ op_mat - op_mat @ H
            total += np.linalg.norm(comm @ psi)**2
        return total

    return f, m


def commutator_cost_number_pres(H, psi, num_electrons, n_orbitals):
    """THIS COULD STILL BE FASTER"""
    pairs = list(combinations(range(n_orbitals), 2))
    m = len(pairs)

    def f(x):
        a = np.sin(x[m]) * np.cos(x[m + 1])
        b = np.sin(x[m]) * np.sin(x[m + 1])
        c = np.cos(x[m])
        U = build_U_from_thetas(n_orbitals, x[:m], pairs)
        symmetries = [(of.FermionOperator(((2 * i, 1), (2 * i, 0)), a)
                       + of.FermionOperator(((2 * i + 1, 1), (2 * i + 1, 0)), b)
                       + (of.FermionOperator(((2 * i, 1), (2 * i, 0)), c)
                       * of.FermionOperator(((2 * i + 1, 1), (2 * i + 1, 0)), 1)))
                      for i in range(n_orbitals)]

        symmetry_inter_ops = [of.get_interaction_operator(s, n_qubits=2 * n_orbitals)
                              for s in symmetries]

        total = 0
        for op in symmetry_inter_ops:
            op.rotate_basis(U.T.conj())
            op = of.get_fermion_operator(op)
            op_mat = of.get_number_preserving_sparse_operator(
                op, num_qubits=2 * n_orbitals,
                num_electrons=num_electrons)
            # print(op_mat.shape)
            # print(H.shape)
            comm = H @ op_mat - op_mat @ H
            total += np.linalg.norm(comm @ psi)**2
        return total

    return f, m


if __name__ == "__main__":

    mol = get_mol("lih", 0.8)

    H_ferm = of.get_fermion_operator(
        mol.get_molecular_hamiltonian())

    H_full = of.get_sparse_operator(H_ferm).todense()
    _, psi_fci = spla.eigsh(H_full, which="SA", k=1)

    H_number = of.get_number_preserving_sparse_operator(
        H_ferm, num_electrons=mol.n_electrons,
        num_qubits=mol.n_orbitals * 2
    ).todense()
    _, psi_fci_number = spla.eigsh(H_number, which="SA", k=1)


    # f, m = commutator_cost(H_full, psi_fci)
    f, m = commutator_cost_number_pres(H_number,
                                       psi_fci_number,
                                       mol.n_electrons,
                                       mol.n_orbitals)
    g = commutator_cost_function_fixed_abc(H_full, psi_fci)

    # x = np.ones(m + 2) * 0.5
    rng = np.random.default_rng(0)
    x = rng.normal(size = m + 2)
    t0 = time.time()
    print(f(x))
    t1 = time.time()
    print(g(x))
    t2 = time.time()
    print(t1 - t0, t2 - t1)