from optimization_different_abc import *

import argparse
from itertools import combinations
import numpy as np
import time

from optimization_abc import variance_restricted, optimize_variance_restricted

def get_mol(molname, bond):
    geometry, description = get_geometry_and_description(molname, bond)

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


def expected_commutator_optimization_results():
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol",
        help="one of the following: lih, h2o, h4_linear, h4_square, h4_rectangle")
    parser.add_argument("bond_min", type=float, help="minimum bond")
    parser.add_argument("bond_max", type=float, help="maximum bond")
    parser.add_argument("--bond_qty", type=int, help="qty of data points",
                        default=1)
    parser.add_argument("--abcmode", default="locked")
    args = parser.parse_args()

    bonds = np.linspace(args.bond_min, args.bond_max, args.bond_qty)

    if args.abcmode == "locked":
        csv_filename = args.mol + "_quasi_symmetry_comm_optimization.csv"
        fieldnames = ["Molecule", "Geometry_Param", "E_FCI", "V_0", "V_optimized",
                      "Sum_CommSq_0", "Sum_CommSq_Optimized", "a_opt", "b_opt", "c_opt"]
    elif args.abcmode == "independent":
        csv_filename = args.mol + "_quasi_symmetry_comm_optimization_different_abc.csv"
        fieldnames = ["Molecule", "Geometry_Param", "E_FCI", "V_0", "V_optimized",
                      "Sum_CommSq_0", "Sum_CommSq_Optimized", "a_opt", "b_opt", "c_opt"]
    else:
        raise ValueError("abcmode must be 'locked' or 'independent'")

    out_data = []
    optimized_parameters = []

    for bond in bonds:
        print(args.mol, bond)
        mol = get_mol(args.mol, bond)

        n_e = mol.n_electrons
        # n_spatial = mol.n_orbitals
        n_qubits = 2 * mol.n_orbitals
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

        pairs = list(combinations(range(mol.n_orbitals), 2))
        m = len(pairs)

        if args.abcmode == "independent":
            print("a, b, c independent for every orbital")
            raise NotImplementedError()

            x_id = np.zeros(m + 2 * mol.n_orbitals)
            # we put constrain a^2 + b^2 + c^2 = 1
            # therefore instead of those we introduce two polar angles per spatial orbital
            for i in range(mol.n_orbitals):
                x_id[m + 2*i] = np.arccos(-2.0 / np.sqrt(6.0))   # phi1_i
                x_id[m + 2*i + 1] = np.pi / 4.0                  # phi2_i


            def f(x):
                thetas, local_abcs = unpack_local_abc_params(x, mol.n_orbitals, m)
                U = build_U_from_thetas(mol.n_orbitals, thetas, pairs)
                total_commutator_norm = 0
                for i in range(mol.n_orbitals):
                    Si_ferm = build_single_local_operator(U, mol.n_orbitals, i, local_abcs)
                    Si_mat = fermion_to_sparse_qubit(Si_ferm, n_qubits)
                    total_commutator_norm += expected_squared_commutator(
                        H_full, Si_mat, psi_full)
                return total_commutator_norm

            print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

            rng = np.random.default_rng()
            x_0 = x_id + rng.normal(scale=1e-3,
                                   size=x_id.shape[0])

            print("before optimization", f(x_0))
            thetas_0, local_abcs_0 = unpack_local_abc_params(
                x_0, mol.n_orbitals, m)
            print("starting abc", local_abcs_0)

            res = minimize(f, x_0,
                           method="L-BFGS-B",
                           options={"maxiter": 100})

            print(res)
            print("after optimization", f(res.x))
            print(res.x)

            thetas_res, local_abcs_res = unpack_local_abc_params(
                res.x, mol.n_orbitals, m)
            print("optimized abc")
            for v in local_abcs_res:
                print(v / v[0])
            print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

        elif args.abcmode == "locked":
            print("a, b, c are the same for all orbitals")
            print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

            gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_statevector(
                psi_full, mol.n_orbitals)

            x_id = np.zeros(m + 2)
            x_id[m] = np.arccos(-2.0 / np.sqrt(6.0))   # c = -2/sqrt(6)
            x_id[m + 1] = np.pi / 4.0                  # a = b = 1/sqrt(6)


            def f(x):
                a = np.sin(x[m]) * np.cos(x[m + 1])
                b = np.sin(x[m]) * np.sin(x[m + 1])
                c = np.cos(x[m])
                U = build_U_from_thetas(mol.n_orbitals, x_id[:m], pairs)
                total_commutator_norm = 0
                for i in range(mol.n_orbitals):
                    Si_ferm = build_single_local_operator(U, mol.n_orbitals, i,
                                                          [(a, b, c)] * mol.n_orbitals)
                    Si_mat = fermion_to_sparse_qubit(Si_ferm, n_qubits)
                    total_commutator_norm += expected_squared_commutator(
                        H_full, Si_mat, psi_full)
                return total_commutator_norm

            rng = np.random.default_rng()
            x_0 = x_id + rng.normal(scale=1e-3,
                                    size=x_id.shape[0])

            print("||[H, S] psi||^2 before optimization", f(x_0))

            res = minimize(f, x_0,
                           method="L-BFGS-B",
                           options={"maxiter": 100})

            print(res.message)
            print("||[H, S] psi||^2 after optimization", f(res.x))
            print(res.x)

            phi1, phi2 = res.x[m], res.x[m + 1]

            # Spherical parameterization for sqrt(a^2 + b^2 + c^2) = 1
            a_opt = np.sin(phi1) * np.cos(phi2)
            b_opt = np.sin(phi1) * np.sin(phi2)
            c_opt = np.cos(phi1)

            print("Optimal abc (rescaled):", 1, b_opt / a_opt, c_opt / a_opt)

            variance_before, _, _, _, _, _ = variance_restricted(
                gamma_a, gamma_b, gamma_ab, x_id, pairs
            )

            variance_after, _, _, _, _, _ = variance_restricted(
                gamma_a, gamma_b, gamma_ab, res.x, pairs
            )

            print("Var S before optimization", variance_before)
            print("Var S after optimization", variance_after)

            out_data.append([args.mol, bond, E_fci, variance_before, variance_after,
                               f(x_0), f(res.x), 1, b_opt / a_opt, c_opt / a_opt])
            optimized_parameters.append(res.x)

            print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
        else:
            raise ValueError("abcmode must be either 'independent' or 'locked'")
    print()
    np.savetxt(csv_filename[:-4] + ".txt",
               np.array(optimized_parameters))

    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, data in enumerate(out_data):
            writer.writerow({fieldnames[j]: data[j] for j in range(len(fieldnames))})
