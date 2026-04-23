"""
For a given problem, we can obtain quasisymmetries optimized
in one way or another. In this script we analyze the properties
of the optimized basis and the optimized quasisymmteries
"""


import numpy as np
import matplotlib.pyplot as plt
import argparse
import openfermion as of

from itertools import combinations, product
from typing import Tuple
from copy import deepcopy
from scipy.optimize import linprog

from chemistry import get_mol, build_U_from_thetas


def diagonal_energies(h_in: of.InteractionOperator,
                      U: np.ndarray,
                      n_up: int,
                      n_down: int) -> Tuple[float, float, float, float]:
    """Rotate h into the basis specified by U and return
    effective energies of its diagonal part:
    lowest energy;
    electron removal energy (highest occupied);
    electron addition energy (lowest empty);
    electron excitation energy (HOMO -> LUMO)"""

    h = deepcopy(h_in)
    h.rotate_basis(U.T.conj())

    one_body_coeffs = np.diag(h.one_body_tensor)

    two_body_coeffs = np.zeros((len(one_body_coeffs), len(one_body_coeffs)))

    for i, j in product(range(len(one_body_coeffs)), repeat=2):
        two_body_coeffs[i, j] = (h.two_body_tensor[i, j, j, i]
                                 - h.two_body_tensor[i, j, i, j])

    print(one_body_coeffs + np.diag(two_body_coeffs))

    e_lowest, ns_lowest = solve_qubo(one_body_coeffs,
                                     two_body_coeffs, n_up, n_down)
    print(e_lowest, ns_lowest)

    lowest_empty = min(np.where(np.isclose(ns_lowest, 0))[0])
    electron_added_state = np.copy(ns_lowest)
    electron_added_state[lowest_empty] = 1
    e_added = (electron_added_state @ one_body_coeffs
               + electron_added_state @ two_body_coeffs @ electron_added_state)
    print(e_added, electron_added_state)

    highest_filled = max(np.where(np.isclose(ns_lowest, 1))[0])
    electron_removed_state = np.copy(ns_lowest)
    electron_removed_state[highest_filled] = 0
    e_removed = (electron_removed_state @ one_body_coeffs
               + electron_removed_state @ two_body_coeffs @ electron_removed_state)
    print(e_removed, electron_removed_state)

    alpha_sites = np.copy(ns_lowest[::2])
    beta_sites = np.copy(ns_lowest[1::2])
    lowest_empty_alpha = min(np.where(np.isclose(alpha_sites, 0))[0])
    highest_filled_alpha = max(np.where(np.isclose(alpha_sites, 1))[0])

    lowest_empty_beta = min(np.where(np.isclose(beta_sites, 0))[0])
    highest_filled_beta = max(np.where(np.isclose(beta_sites, 1))[0])

    alpha_sites[lowest_empty_alpha] = 1
    alpha_sites[highest_filled_alpha] = 0

    alpha_excited_state = np.zeros_like(ns_lowest)
    for i in range(len(ns_lowest) // 2):
        alpha_excited_state[2 * i] = alpha_sites[i]
        alpha_excited_state[2 * i + 1] = beta_sites[i]

    e_alpha_excited = (alpha_excited_state @ one_body_coeffs
               + alpha_excited_state @ two_body_coeffs @ alpha_excited_state)

    print(e_alpha_excited, alpha_excited_state)

    alpha_sites = ns_lowest[::2]
    beta_sites[lowest_empty_beta] = 1
    beta_sites[highest_filled_beta] = 0
    beta_excited_state = np.zeros_like(ns_lowest)
    for i in range(len(ns_lowest) // 2):
        beta_excited_state[2 * i] = alpha_sites[i]
        beta_excited_state[2 * i + 1] = beta_sites[i]


    e_beta_excited = (beta_excited_state @ one_body_coeffs
                       + beta_excited_state @ two_body_coeffs @ beta_excited_state)
    print(e_beta_excited, beta_excited_state)

    return e_lowest, e_removed, e_added, min(e_alpha_excited, e_beta_excited)


def solve_qubo(linear_term: np.ndarray,
               quadratic_term: np.ndarray,
               n_up: int,
               n_down: int) -> Tuple[float, np.ndarray]:
    """Given a QUBO problem, return the optimal value and the optimal x.
    We first convert the problem to a constrained integer linear programming problem,
    then solve it with linprog

    n_up, n_down are used for constraints"""

    linear_effective = linear_term + np.diag(quadratic_term)
    upper_triangular = np.triu(quadratic_term + quadratic_term.T, 1)
    mask = np.ones_like(upper_triangular)
    mask = np.triu(mask, 1)
    nz = np.nonzero(mask)
    nz_flat = np.array([upper_triangular[nz[0][i], nz[1][i]]
                        for i in range(len(nz[0]))])
    cost_vector = np.concatenate((linear_effective, nz_flat))

    constraints = []
    for z_count in range(len(nz[0])):
        i = nz[0][z_count]
        j = nz[1][z_count]

        row_lt_i = np.zeros_like(cost_vector)
        row_lt_i[i] = -1
        row_lt_i[len(linear_term) + z_count] = 1

        row_lt_j = np.zeros_like(cost_vector)
        row_lt_j[j] = -1
        row_lt_j[len(linear_term) + z_count] = 1

        row_gt_sum = np.zeros_like(cost_vector)
        row_gt_sum[i] = 1
        row_gt_sum[j] = 1
        row_gt_sum[len(linear_term) + z_count] = -1

        constraints.append(row_lt_i)
        constraints.append(row_lt_j)
        constraints.append(row_gt_sum)

    A_ub = np.array(constraints)
    b_ub = np.array([0, 0, 1] * len(nz[0]))

    A_eq = np.zeros((2, len(cost_vector)))
    for i in range(len(linear_term) // 2):
        A_eq[0, 2 * i] = 1
        A_eq[1, 2 * i + 1] = 1

    b_eq = [n_up, n_down]

    res = linprog(cost_vector, A_ub, b_ub, A_eq, b_eq,
                  bounds=(0, 1), integrality=1)

    return res.fun, res.x[:len(linear_term)]


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol",
        help="one of the following: lih, h2o, h4_linear, h4_square, h4_rectangle")
    parser.add_argument("bond", type=float, help="bond")
    parser.add_argument("x",
                        help="path to file with the optimized S_i data")

    args = parser.parse_args()
    if args.mol in ("h2", "lih", "h2o", "h4_linear", "h4_square", "h4_rectangle"):
        mol = get_mol(args.mol, args.bond)
        h = mol.get_molecular_hamiltonian()

    U = build_U_from_thetas(
        mol.n_orbitals,
        np.loadtxt(args.x),
    list(combinations(range(mol.n_orbitals), 2))
    )

    diagonal_energies(h, U, mol.n_electrons // 2,
                      mol.n_electrons // 2 + mol.n_electrons % 2)

    w, v = np.linalg.eig(U)
    print(np.angle(w))

