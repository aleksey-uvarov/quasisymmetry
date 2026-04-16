from optimization_different_abc import *

import argparse
from itertools import combinations
import numpy as np
import time

def get_mol(molname, bond):
    geometry, description = get_geometry_and_description(args.mol, args.bond)

    mol = MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        description=description
    )
    mol = run_pyscf(mol, run_scf=True, run_fci=False, run_cisd=False)

    return mol


def expected_squared_commutator(H: np.ndarray,
                                S: np.ndarray,
                                psi: np.ndarray) -> float:
    Apsi = H.dot(S.dot(psi)) - S.dot(H.dot(psi))
    return np.linalg.norm(Apsi)**2


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol",
        help="one of the following: lih, h2o, h4_linear, h4_square, h4_rectangle")
    parser.add_argument("bond", type=float, help="bond length")
    args = parser.parse_args()

    mol = get_mol(args.mol, args.bond)

    n_e = mol.n_electrons
    n_spatial = mol.n_orbitals
    n_qubits = 2 * n_spatial
    dim = 1 << n_qubits

    H_ferm = get_fermion_operator(mol.get_molecular_hamiltonian())
    H_qubit = jordan_wigner(H_ferm)
    H_full = get_sparse_operator(H_qubit, n_qubits).tocsc()

    # Fixed-N basis
    basis_bitstrings = [b for b in range(dim) if popcount(b) == n_e]
    basis_idx = np.array(basis_bitstrings, dtype=int)

    H_sub = H_full[basis_idx, :][:, basis_idx].tocsc()
    evals, evecs = spla.eigsh(H_sub, k=1, which="SA")
    E_fci = float(np.real(evals[0]))
    v_sub = evecs[:, 0]

    # Full-space FCI state
    psi_full = np.zeros(dim, dtype=np.complex128)
    psi_full[basis_idx] = v_sub
    psi_full /= np.linalg.norm(psi_full)

    pairs = list(combinations(range(n_spatial), 2))
    m = len(pairs)

    x_id = np.zeros(m + 2 * n_spatial)
    # we put constrain a^2 + b^2 + c^2 = 1
    # therefore instead of those we introduce two polar angles per spatial orbital
    for i in range(n_spatial):
        x_id[m + 2*i] = np.arccos(-2.0 / np.sqrt(6.0))   # phi1_i
        x_id[m + 2*i + 1] = np.pi / 4.0                  # phi2_i


    def f(x):
        thetas, local_abcs = unpack_local_abc_params(x, n_spatial, m)
        U = build_U_from_thetas(n_spatial, thetas, pairs)
        total_commutator_norm = 0
        for i in range(n_spatial):
            Si_ferm = build_single_local_operator(U, n_spatial, i, local_abcs)
            Si_mat = fermion_to_sparse_qubit(Si_ferm, n_qubits)
            total_commutator_norm += expected_squared_commutator(
                H_full, Si_mat, psi_full)
        return total_commutator_norm

    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    print("before optimization", f(x_id))
    thetas_0, local_abcs_0 = unpack_local_abc_params(
        x_id, n_spatial, m)
    print("starting abc", local_abcs_0)

    rng = np.random.default_rng()

    res = minimize(f,
                   x_id + rng.normal(scale=1e-4,
                                              size=x_id.shape[0]),
                   method="L-BFGS-B",
                   options={"maxiter": 100})

    print(res)
    print("after optimization", f(res.x))
    print(res.x)

    thetas_res, local_abcs_res = unpack_local_abc_params(
        res.x, n_spatial, m)
    print("optimized abc")
    for v in local_abcs_res:
        print(v / v[0])
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))



