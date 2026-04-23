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
                      U: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotate h into the basis specified by U and return
    effective energies of its diagonal part:
    lowest energy;
    electron removal energy (HOMO);
    electron addition energy (LUMO);
    electron excitation energy (HOMO -> LUMO)"""

    h = deepcopy(h_in)
    h.rotate_basis(U.T.conj())

    one_body_coeffs = np.diag(h.one_body_tensor)
    print(one_body_coeffs)

    two_body_coeffs = np.zeros((len(one_body_coeffs), len(one_body_coeffs)))

    for i, j in product(range(len(one_body_coeffs)), repeat=2):
        two_body_coeffs[i, j] = h.two_body_tensor[i, j, j, i] - h.two_body_tensor[i, j, i, j]

    e_lowest, ns_lowest = solve_qubo(one_body_coeffs, two_body_coeffs)
    print(e_lowest, ns_lowest)

    return e_lowest, None, None, None


def solve_qubo(linear_term: np.ndarray,
               quadratic_term: np.ndarray) -> Tuple[float, np.ndarray]:
    """Given a QUBO problem, return the optimal value and the optimal x.
    We first convert the problem to a constrained integer linear programming problem,
    then solve it with linprog"""

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

    res = linprog(cost_vector, A_ub, b_ub, bounds=(0, 1), integrality=1)

    return res.fun, res.x[:len(linear_term)]


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol",
        help="one of the following: lih, h2o, h4_linear, h4_square, h4_rectangle")
    parser.add_argument("bond", type=float, help="bond")
    parser.add_argument("x",
                        help="path to file with the optimized S_i data")

    args = parser.parse_args()
    mol = get_mol(args.mol, args.bond)
    h = mol.get_molecular_hamiltonian()

    U = build_U_from_thetas(
        mol.n_orbitals,
        np.loadtxt(args.x),
    list(combinations(range(mol.n_orbitals), 2))
    )

    diagonal_energies(h, U)

