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

from chemistry import get_mol, build_U_from_thetas


def diagonal_energies(h_in: of.InteractionOperator,
                      U: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotate h into the basis specified by U and return
    effective energies of its diagonal part:
    loweest energy;
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


    return None, None, None, None



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

