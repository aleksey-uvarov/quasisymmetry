from optimization_different_abc import *

import argparse
from itertools import combinations
import numpy as np
import time
import openfermion as of
from pathlib import Path

from optimization_abc import variance_restricted

from chemistry import get_mol

def expected_squared_commutator(H: np.ndarray,
                                S: np.ndarray,
                                psi: np.ndarray) -> float:
    Apsi = H.dot(S.dot(psi)) - S.dot(H.dot(psi))
    return np.linalg.norm(Apsi)**2


def expected_squared_commutator_dm(H: np.ndarray,
                                S: np.ndarray,
                                rho: np.ndarray) -> float:
    commutator = H @ S - S @ H
    c_dag_c = commutator.T.conj() @ commutator
    return np.trace(c_dag_c @ rho)


def commutator_cost_function_fixed_abc(H_full, psi_full):
    n_spin_orbitals = int(np.log2(psi_full.shape[0]))
    n_orbitals = n_spin_orbitals // 2
    pairs = list(combinations(range(n_orbitals), 2))
    m = len(pairs)
    def f(x):
        a = np.sin(x[m]) * np.cos(x[m + 1])
        b = np.sin(x[m]) * np.sin(x[m + 1])
        c = np.cos(x[m])
        U = build_U_from_thetas(n_orbitals, x[:m], pairs)
        total_commutator_norm = 0
        for i in range(n_orbitals):
            Si_ferm = build_single_local_operator(U, n_orbitals, i,
                    [(a, b, c)] * n_orbitals)
            # Si_ferm = normal_ordered(
            #     rotated_seniority_orbital_fermion(
            #         U, i, n_orbitals, a, b, c
            #     )
            # )
            Si_mat = fermion_to_sparse_qubit(Si_ferm, n_spin_orbitals)
            total_commutator_norm += expected_squared_commutator(
                H_full, Si_mat, psi_full)
        return total_commutator_norm
    return f


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


def sum_of_variances_cost_function_fixed_abc(psi_full):
    n_spin_orbitals = int(np.log2(psi_full.shape[0]))
    n_orbitals = n_spin_orbitals // 2
    pairs = list(combinations(range(n_orbitals), 2))
    gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_statevector(
        psi_full, n_orbitals)
    def f(x):
        return variance_restricted(gamma_a, gamma_b, gamma_ab, x, pairs)[0]

    return f


def callback(intermediate_result):
    print(intermediate_result.fun)
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol",
        help="one of the following: lih, h2o, h4_linear, h4_square, h4_rectangle, h2")
    parser.add_argument("bond", type=float, help="bond")
    # parser.add_argument("--optruns", type=int, default=1)
    parser.add_argument("initialguesses",
                        help="path to file with initial guesses (one line = one point)")
    # parser.add_argument("--noise_power", type=int, default=-4)

    args = parser.parse_args()
    mol = get_mol(args.mol, args.bond)

    print("a, b, c are the same for all orbitals")
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

    H_ferm = of.get_fermion_operator(
        mol.get_molecular_hamiltonian())
    H_qubit = jordan_wigner(H_ferm)
    H_full = get_sparse_operator(H_qubit, 2 * mol.n_orbitals).tocsc()

    basis_bitstrings = [b for b in range(2**(2 * mol.n_orbitals))
                        if popcount(b) == mol.n_electrons]
    basis_idx = np.array(basis_bitstrings, dtype=int)

    H_sub = H_full[basis_idx, :][:, basis_idx].tocsc()
    evals, evecs = spla.eigsh(H_sub, k=1, which="SA")
    E_fci = float(np.real(evals[0]))
    v_sub = evecs[:, 0]

    # Full-space FCI state
    psi_full = np.zeros(2**(2 * mol.n_orbitals), dtype=np.complex128)
    psi_full[basis_idx] = v_sub
    psi_full /= np.linalg.norm(psi_full)

    H_number = of.get_number_preserving_sparse_operator(
        H_ferm, num_electrons=mol.n_electrons,
        num_qubits=mol.n_orbitals * 2
    ).todense()
    _, psi_fci_number = spla.eigsh(H_number, which="SA", k=1)

    f = sum_of_variances_cost_function_fixed_abc(psi_full)

    g, m = commutator_cost_number_pres(H_number,
                                       psi_fci_number,
                                       mol.n_electrons,
                                       mol.n_orbitals)

    initial_guesses = np.loadtxt(args.initialguesses)

    n_points = initial_guesses.shape[0]

    fieldnames = ["E_FCI", "V_0", "V_optimized",
     "Sum_CommSq_0", "Sum_CommSq_Optimized", "a_opt", "b_opt", "c_opt"]

    with open(args.mol + "_" + str(args.bond) +  "_"
                  + args.initialguesses + "_var_opt_data.txt",
              "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

    for i in range(n_points):
        x_0 = initial_guesses[i, :]
        print("x0", x_0)
        res = minimize(f, x_0,
                       method="L-BFGS-B",
                       options={"maxiter": 100},
                       callback=callback)
        print(res.message)

        variance_before = f(x_0)
        variance_after = res.fun

        phi1, phi2 = res.x[m], res.x[m + 1]

        # Spherical parameterization for sqrt(a^2 + b^2 + c^2) = 1
        a_opt = np.sin(phi1) * np.cos(phi2)
        b_opt = np.sin(phi1) * np.sin(phi2)
        c_opt = np.cos(phi1)

        out_row = np.array([E_fci, variance_before, variance_after,
                               g(x_0), g(res.x), 1, b_opt / a_opt, c_opt / a_opt])

        with open(args.mol + "_" + str(args.bond) + "_"
                  + args.initialguesses + "_x_var_opt.txt", "ab") as fp:
            np.savetxt(fp, res.x.reshape(1, res.x.shape[0]))

        with open(args.mol + "_" + str(args.bond) +  "_"
                  + args.initialguesses + "_var_opt_data.txt", "ab") as fp:
            np.savetxt(fp, out_row.reshape(1, out_row.shape[0]))