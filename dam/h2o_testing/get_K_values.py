import argparse
import csv
import sys
import pyscf
import numpy as np
from scipy.special import binom
sys.path.append('../')
import cluster_number, cost_functions, utils
sys.path.append('../../')
from chemistry import get_geometry_and_description
import ffsim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from optimize_symmetries import parities, get_fci
from optimize_symmetries import parity_matrix_to_quasisymmetries, x_to_rotation, commutator_cost, variance_cost # improved to cluster_number.commutator_cost_v2
import scipy.optimize
from math import comb
from metrics import subspace_matrix, orthogonalize_degenerate
from chemistry import CHEMICAL_PRECISION
from math import comb

# define examples of cluster matrices
example_cluster_matrices = []

# choice 0: each cluster = 1 orbital; orbitals 0-2 -> numbers; orbitals 3-5 -> parities
example_cluster_number_matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0]
])
example_cluster_parity_matrix = np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0]
])
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))

# choice 1: some cluster numbers, with |cluster| >= 2 orbitals; no cluster parities
example_cluster_number_matrix = np.array([
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 1, 0]
])
example_cluster_parity_matrix = np.array([])
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))

# choice 2: choice 1, but switching number and parity matrices
example_cluster_number_matrix = np.array([])

example_cluster_parity_matrix = np.array([
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 1, 0]
])
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))

# choice 3: some pairs of orbitals whose corresponding heat map tile is darkest; and one parity cluster
example_cluster_number_matrix = np.array([
    [0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0]
])
example_cluster_parity_matrix = np.array([
    [1, 1, 1, 0, 0, 0, 0]
])
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))

# choice 4: all one-orb numbers, taken singularly (-> no cluster parities)
example_cluster_number_matrix = np.eye(7)
example_cluster_parity_matrix = np.array([])
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))

# choice 5 (QSENSE): no cluster numbers, all one-orb parities, taken singularly
example_cluster_number_matrix = np.array([]) 
example_cluster_parity_matrix = np.eye(7)
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))

# choice 6: numbers for pairs of orbitals whose corresponding heat map tile is darkest; no parities
example_cluster_number_matrix = np.array([
    [1, 0, 0, 0, 1, 0, 0,],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 0]
])
example_cluster_parity_matrix = np.array([])
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))

# choice 7: same as above, but only orbital-pair parities
example_cluster_number_matrix = np.array([])
example_cluster_parity_matrix = np.array([
    [1, 0, 0, 0, 1, 0, 0,],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 0]
])
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))

# choice 8: only parities, random
example_cluster_number_matrix = np.array([])
example_cluster_parity_matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0,],
    [0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0]
])
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))


# choice 9: only one big parity
example_cluster_number_matrix = np.array([])
example_cluster_parity_matrix = np.array([
    [1, 1, 1, 0, 0, 0, 0,]
])
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))

# choice 10: overload of parities
example_cluster_number_matrix = np.array([])
example_cluster_parity_matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0,],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 0, 0, 0,],
    [0, 1, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 0]
])
example_cluster_matrices.append((example_cluster_number_matrix, example_cluster_parity_matrix))

def compute_K_values(score_type, example_cluster_matrices_index, bond_length, hoh_angle_deg):
    """This function reproduces workflow_cluster_numbers_and_parities.ipynb"""
    molecule = 'h2o'
    geometry, description = get_geometry_and_description(molecule, bond_length, hoh_angle_deg=hoh_angle_deg)
    mol = pyscf.M()
    mol.build(atom=geometry, basis="sto-3g", verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.chkfile = f"hamiltonians/{description}.chk"  # Will save to this file
    mf.kernel() 
    norb = mol.nao
    nelec = mol.nelec
    moldata = pyscf.lib.chkfile.load_mol(mf.chkfile)
    mf_update = pyscf.scf.RHF(mol) # why again?
    mf_update.update_from_chk(mf.chkfile)
    moldata_ffsim = ffsim.MolecularData.from_scf(mf_update)
    dumpdata = { # (dump means save/write out)
        "NORB": mol.nao,
        "NELEC": mol.nelec,
        "H1": mf.mo_coeff.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ mf.mo_coeff,
        "H2": pyscf.ao2mo.full(mol, mf.mo_coeff),
        "ECORE": mol.energy_nuc()
    }
    e_fci, fci_state = get_fci(dumpdata)
    one_orb_num_operators = cluster_number.build_one_orb_num_operators(norb, nelec)
    two_orb_num_operators = []
    for i in range(norb):
        for j in range(i+1, norb):
            two_orb_num_operators.append(one_orb_num_operators[i] + one_orb_num_operators[j])
    one_orb_expnum_operators = [cluster_number.from_num_operator_to_expnum_operator(op, 2) for op in one_orb_num_operators]
    two_orb_expnum_operators = [cluster_number.from_num_operator_to_expnum_operator(op, 4) for op in two_orb_num_operators]
    # parities
    one_orb_par_operators = parities(norb, nelec) # Aleksey's implementation
    two_orb_par_operators = []
    for i in range(norb):
        for j in range(i+1, norb):
            two_orb_par_operators.append(one_orb_par_operators[i] @ one_orb_par_operators[j])
    h = ffsim.linear_operator(moldata_ffsim.hamiltonian, norb, nelec)
    cluster_number_matrix, cluster_parity_matrix = example_cluster_matrices[example_cluster_matrices_index]
    cluster_number_operators = cluster_number.number_matrix_to_operators(
        cluster_number_matrix, norb, nelec, expnum=False)

    cluster_parity_operators = parity_matrix_to_quasisymmetries( # Aleksey's implementation
        cluster_parity_matrix, norb, nelec)
    quasisymmetry_operators = cluster_number_operators + cluster_parity_operators
    if score_type == 'noncommutativity':
        # Cost = sum over symmetries S_k of ||[H(U), S_k]|Ψ(U)⟩||²
        f = cost_functions.commutator_cost_v2(moldata_ffsim, quasisymmetry_operators, fci_state) # commutator_cost builds the cost function f: x upper triangle of antisym matrix |-> \sum_k ||[H(rotated with exp(x)), S_k]|Ψ(rotated with exp(x)⟩||²
    if score_type == 'variance':
        # sym:variance_cost
        f = cost_functions.variance_cost_beyond_parities(moldata_ffsim, quasisymmetry_operators, fci_state)
    if score_type == 'eval_eq':
        cluster_number_evals = [round(np.real(fci_state.T.conj() @ (op @ fci_state))) for op in cluster_number_operators] # best guess of eigenvalues
        cluster_parity_evals = [np.sign(fci_state.T.conj() @ (op @ fci_state)) for op in cluster_parity_operators] # same
        evals = cluster_number_evals + cluster_parity_evals
        f = cost_functions.eval_eq_cost(quasisymmetry_operators, evals, fci_state, norb, nelec) 

    # --- Step 3.4: Initial guess (identity rotation) ---
    x0 = np.zeros(comb(norb, 2))
    initial_cost = f(x0)

    result = scipy.optimize.minimize(
        f, x0,
        method='L-BFGS-B',
        options={'maxiter': 30, 'disp': True}
    )
    optimized_cost = result.fun
    x_opt = result.x
    U_opt = x_to_rotation(x_opt, norb)
    rotated_h = moldata_ffsim.hamiltonian.rotated(U_opt)
    rotated_h_linop = ffsim.linear_operator(rotated_h, norb, nelec)
    rotated_fci_state = ffsim.apply_orbital_rotation(
        fci_state, U_opt, norb, nelec)

    sectors = cluster_number.number_and_parity_symmetry_sectors(cluster_number_matrix, cluster_parity_matrix, norb, nelec)

    state_projections_in_sectors = {} # key = sector label (as in sectors), value = (projection of rotated_fci_state into sectors, norm squared)
    for sector_label, sector_indices in sectors.items():
        projection = np.zeros(rotated_fci_state.shape, dtype='complex')
        projection[sector_indices] = rotated_fci_state[sector_indices]
        norm_squared = np.linalg.norm(projection)**2
        energy = np.real(projection.T.conj() @ rotated_h_linop @ projection / norm_squared) if norm_squared > 0 else np.nan
        state_projections_in_sectors[sector_label] = (projection, norm_squared, energy)

    ordered_state_projections_in_sectors = sorted(state_projections_in_sectors.items(), key=lambda x: x[1][1], reverse=True)

    def projected_energy_sectors(K_sector):
        """Compute energy using only the K_sector most important sectors"""
        # Create compressed coefficient vector
        compressed_coeffs = np.zeros_like(rotated_fci_state, dtype='complex')
        for i in range(K_sector):
            sector_label, (projection, norm_squared, energy) = ordered_state_projections_in_sectors[i]
            compressed_coeffs += projection

        # Normalize
        compressed_coeffs /= np.linalg.norm(compressed_coeffs)

        # Compute energy
        e_proj = compressed_coeffs.T.conj() @ rotated_h_linop @ compressed_coeffs
        return e_proj.real

    # Convergence in the number of sector states retained
    K_sectors_values = []
    K_sectors_energies = []
    for K_sectors in range(1, len(ordered_state_projections_in_sectors) + 1):
        K_sectors_values.append(K_sectors)
        e_K = projected_energy_sectors(K_sectors)
        K_sectors_energies.append(e_K)
        error = e_K - e_fci
        if error < CHEMICAL_PRECISION:
            break

    sector_hamiltonians = {}
    for sector_label, sector_indices in sectors.items():
        sector_hamiltonians[sector_label] = subspace_matrix( # should still be able to reuse Aleksey's subspace_matrix here!
            rotated_h_linop, sector_indices)   

    sector_energies = {}
    sector_states = {}
    for label, h_sub in sector_hamiltonians.items():
        # Get all eigenvalues (full diagonalization for small systems)
        w, v = np.linalg.eigh(h_sub)
        v_orth = orthogonalize_degenerate(w, v)
        sector_energies[label] = w
        sector_states[label] = v_orth

    sector_ground_energies = {label: energies[0] for label, energies in sector_energies.items()}
    global_min_sector = min(sector_ground_energies, key=sector_ground_energies.get)
    e_decoupled = sector_ground_energies[global_min_sector]

    full_space_vectors = []
    sector_labels_list = []
    for label, indices in sectors.items():
        # Get the states for this sector
        v_sector = sector_states[label]
        n_states = v_sector.shape[1]

        # Create full-space vectors (zeros everywhere except in this sector)
        vectors_in_sector = np.zeros((rotated_h_linop.shape[0], n_states),
                                    dtype='complex')
        vectors_in_sector[indices, :] = v_sector
        full_space_vectors.append(vectors_in_sector)

        # Track which sector each state belongs to
        for i in range(n_states):
            sector_labels_list.append((label, i))

    full_space_vectors_cat = np.concatenate(full_space_vectors, axis=1)

    coefficients = full_space_vectors_cat.T.conj() @ rotated_fci_state

    sorted_indices = np.argsort(np.abs(coefficients))[::-1]

    # Function to compute projected energy using top K states; see metrics.py; rewritten here for clarity
    def projected_energy_states(K_states):
        """Compute energy using only the K_states most important sector eigenstates"""
        # Create compressed coefficient vector
        compressed_coeffs = np.zeros_like(coefficients, dtype='complex')
        compressed_coeffs[sorted_indices[:K_states]] = coefficients[sorted_indices[:K_states]]

        # Normalize
        compressed_coeffs /= np.linalg.norm(compressed_coeffs)

        # Projected state in same representation as as rotated_fci_state
        projected_state = full_space_vectors_cat @ compressed_coeffs
        projected_state /= np.linalg.norm(projected_state) # redundant

        # Compute energy
        e_proj = projected_state.T.conj() @ rotated_h_linop @ projected_state
        return e_proj.real

    K_states_values = []
    K_states_energies = []
    for K_states in range(1, len(coefficients) + 1):
        K_states_values.append(K_states)
        e_K = projected_energy_states(K_states)
        K_states_energies.append(e_K)
        error = e_K - e_fci
        if error < CHEMICAL_PRECISION:
            break

    return K_sectors, K_states


def run_single(score_type, example_cluster_matrices_index, bond_length, hoh_angle_deg, output_file):
    K_sectors, K_states = compute_K_values(
        score_type, example_cluster_matrices_index, bond_length, hoh_angle_deg
    )

    indices_both = [0, 3]
    indices_number = [1, 4, 6]
    indices_parity = [2, 5, 7, 8, 9, 10]
    if example_cluster_matrices_index in indices_both:
        num_or_par = 'both'
    if example_cluster_matrices_index in indices_number:
        num_or_par = 'N'
    if example_cluster_matrices_index in indices_parity:
        num_or_par = 'P'

    with open(output_file, "a") as f:
        f.write(
            "Input: "
            f"score_type = {score_type}, "
            f"example_cluster_matrices_index = {example_cluster_matrices_index}, "
            f"bond_length = {bond_length}, "
            f"hoh_angle_deg = {hoh_angle_deg}\n"
            f"numbers or parities: {num_or_par}\n"
            f"K_sectors = {K_sectors}\n"
            f"K_states = {K_states}\n"
            "\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_csv", default="params.csv")
    args = parser.parse_args()

    with open(args.params_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_single(
                score_type=row["score_type"],
                example_cluster_matrices_index=int(row["example_cluster_matrices_index"]),
                bond_length=float(row["bond_length"]),
                hoh_angle_deg=float(row["hoh_angle_deg"]),
                output_file=row["output_file"],
            )


if __name__ == "__main__":
    main()