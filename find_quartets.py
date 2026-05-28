"""Find approximate symmetries in the form of U * s_pq * U^*,
where s_pq is a parity of population of orbitals p, q"""

import argparse
import numpy as np
import time
import ffsim
import scipy
import pyscf
import networkx as nx
import matplotlib.pyplot as plt
import jax

from functools import cache
from matplotlib.colors import LogNorm
from math import comb

from tqdm import tqdm

from optimize import x_to_rotation
from xs_to_metrics import subspace_matrix

@cache
def make_quartets(norb: int, nelec):
    """pair parity operators s_i s_j. for i=j, just s_i"""
    quartets = {}
    local_parities = []
    for i in range(norb):
        s_alpha = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(i)): -2,
                (): 1
            }
        )
        s_beta = ffsim.FermionOperator(
            {
                (ffsim.cre_b(i), ffsim.des_b(i)): -2,
                (): 1
            }
        )
        s = s_alpha * s_beta
        local_parities.append(ffsim.linear_operator(s, norb, nelec))
    for i in range(norb):
        for j in range(i, norb):
            if i == j:
                quartets[(i, j)] = local_parities[i]
            else:
                quartets[(i, j)] = local_parities[i] @ local_parities[j]
    return quartets


def all_quartet_commutators(moldata: ffsim.MolecularData, state, U):
    """Return a weighted graph with edge weights being the commmutator norms.
    Self-edges denote single-site parity commutators"""
    G = nx.complete_graph(moldata.norb)
    quartets = make_quartets(moldata.norb, moldata.nelec)

    rotated_h = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                      norb=moldata.norb, nelec=moldata.nelec)
    rotated_state = ffsim.apply_orbital_rotation(state, U, moldata.norb, moldata.nelec)

    for i in range(moldata.norb):
        for j in range(i, moldata.norb):
            nc_factor = quartet_noncommutation_factor(rotated_h, rotated_state, i, j)
            if i != j:
                G[i][j]['weight'] = nc_factor
            else:
                G.add_edge(i, i, weight=nc_factor)

    return G


def quartet_noncommutation_factor(h,
                       state: np.ndarray,
                       i: int,
                       j: int):
    quartets = make_quartets(moldata.norb, moldata.nelec)
    commutator = h @ quartets[(i, j)] - quartets[(i, j)] @ h
    state_after_commutator = commutator @ state
    return np.linalg.norm(state_after_commutator)**2


def visualize_nc(moldata, state, U, mo=True, save=False):
    G = all_quartet_commutators(moldata,
                                state,
                                U)

    print("NC factors")
    # for e in G.edges(data=True):
    #     print(e[0], e[1], e[2]['weight'])

    if mo:
        plt.figure()
        plt.imshow(mf.mo_coeff @ U, cmap="PuOr", vmin=-1, vmax=1)
        plt.yticks(range(mol.nao), mol.ao_labels())
        plt.xticks(range(mol.nao), range(mol.nao))
        plt.title("Optimized orbitals \n" + args.molpath)
        plt.colorbar()
        if save:
            plt.savefig("canonical_orbitals.png", dpi=600, bbox_inches="tight", format="png")

    adj = np.triu(nx.to_numpy_array(G), 0)
    plt.figure()
    plt.imshow(adj, norm=LogNorm(vmin=1e-4, vmax=1))
    plt.colorbar()
    plt.xticks(range(mol.nao), range(mol.nao))
    plt.yticks(range(mol.nao), range(mol.nao))
    if args.reference == "fci":
        plt.title("Quartet noncommutativity norm $||[H, s_{pq}]|FCI\\rangle||^2$ \n" + args.molpath
                  + "\n Diagonal entries are $||[H, s_{p}]|FCI\\rangle||^2$")

        if save:
            plt.savefig("quartets.png", dpi=600, bbox_inches="tight", format="png")
    elif args.reference == "hf":
        plt.title("Quartet noncommutativity norm $||[H, s_{pq}]|HF\\rangle||^2$ \n" + args.molpath
                  + "\n Diagonal entries are $||[H, s_{p}]|HF\\rangle||^2$")
        if save:
            plt.savefig("quartets_hf.png", dpi=600, bbox_inches="tight", format="png")
    plt.show()


def nc_cost(moldata, state, quartet_graph):
    def f(x):
        U = x_to_rotation(x, moldata.norb)
        h = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                      norb=moldata.norb, nelec=moldata.nelec)
        rotated_state = ffsim.apply_orbital_rotation(state, U, moldata.norb, moldata.nelec)
        total_nc = 0
        for (i, j) in quartet_graph.edges():
            total_nc += quartet_noncommutation_factor(h, rotated_state, i, j)
        return total_nc
    return f


def quartet_sectors(quartets, norb, nelec):
    dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
    bitstrings = ffsim.addresses_to_strings(range(dim), norb, nelec,
                                            bitstring_type=ffsim.BitstringType.INT, concatenate=False)

    quartet_bit_masks = []
    for q in quartets:
        quartet_bit_masks.append(2 ** q[0] + 2 ** q[1])

    sectors = {}
    for i in range(state.shape[0]):
        ab_parities = bitstrings[0][i] ^ bitstrings[1][i]
        sector_label = tuple(
            (int.bit_count(int(ab_parities & q)) % 2 for q in quartet_bit_masks)
        )
        sectors.setdefault(sector_label, []).append(i)

    return sectors


def sector_metrics(moldata, state, U,
                   sectors,
                   target_energy,
                   max_states_per_sector=100):
    print("sector dimensions:")
    print({k: len(v) for k, v in sectors.items()})


    rotated_h_linop = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                            norb=moldata.norb, nelec=moldata.nelec)
    rotated_state = ffsim.apply_orbital_rotation(state, U,
                                                 moldata.norb, moldata.nelec)

    sector_hamiltonians = {k: subspace_matrix(rotated_h_linop, v)
                           for k, v in sectors.items()}

    sector_eigen_decompositions = {}
    for sector_label, h_local in sector_hamiltonians.items():
        if h_local.shape[0] <= max_states_per_sector:
            sector_eigen_decompositions[sector_label] = np.linalg.eigh(h_local)
        else:
            sector_eigen_decompositions[sector_label] = scipy.sparse.linalg.eigsh(
                h_local, which="SA", k=max_states_per_sector)

    pooled_energies = {}
    for sector_label in sectors.keys():
        for i in range(len(sector_eigen_decompositions[sector_label][0])):
            pooled_energies[(sector_label, i)] = (
                sector_eigen_decompositions[sector_label][0][i]
            )

    vector_labels = list(pooled_energies.keys())
    energy_order = np.argsort(list(pooled_energies.values()))

    # pulling vectors by their local energies
    current_state_vectors = np.array(())
    for i in tqdm(range(len(energy_order))):
        sector_label = vector_labels[energy_order[i]][0]
        vector_id_in_sector = vector_labels[energy_order[i]][1]
        next_vector = sector_eigen_decompositions[sector_label][1][:, vector_id_in_sector]
        next_vector_big = np.zeros(rotated_h_linop.shape[0], dtype="complex")
        next_vector_big[sectors[sector_label]] = next_vector

        if i == 0:
            current_state_vectors = next_vector_big
        else:
            current_state_vectors = np.vstack([current_state_vectors, next_vector_big])

        h_subspace = current_state_vectors.conj() @ rotated_h_linop @ current_state_vectors.T

        if i == 0:
            subspace_energy = h_subspace.real
        elif i == 1:
            w, v = np.linalg.eigh(h_subspace)
            subspace_energy = w[0]
        else:
            subspace_energy = scipy.sparse.linalg.eigsh(h_subspace, k=1, which="SA")[0][0]


        if subspace_energy < target_energy:
            print("K_en = {0:}".format(i + 1))
            K_en =  i + 1
            break
    else:
        K_en = np.nan
        print("Chemical accuracy not reached, try more vectors per sector")

    pooled_overlaps = {}
    for sector_label in sectors.keys():
        for i in range(len(sector_eigen_decompositions[sector_label][0])):
            vector_id_in_sector = vector_labels[energy_order[i]][1]
            vector = sector_eigen_decompositions[sector_label][1][:, vector_id_in_sector]
            projected_reference = state[sectors[sector_label]]
            pooled_overlaps[(sector_label, i)] = abs(vector.T.conj() @ projected_reference)**2

    overlap_order = np.argsort(list(pooled_overlaps.values()))

    # pulling vectors by their local energies
    current_state_vectors = np.array(())
    for i in tqdm(range(len(overlap_order))):
        sector_label = vector_labels[energy_order[i]][0]
        vector_id_in_sector = vector_labels[energy_order[i]][1]
        next_vector = sector_eigen_decompositions[sector_label][1][:, vector_id_in_sector]
        next_vector_big = np.zeros(rotated_h_linop.shape[0], dtype="complex")
        next_vector_big[sectors[sector_label]] = next_vector

        if i == 0:
            current_state_vectors = next_vector_big
        else:
            current_state_vectors = np.vstack([current_state_vectors, next_vector_big])

        h_subspace = current_state_vectors.conj() @ rotated_h_linop @ current_state_vectors.T

        if i == 0:
            subspace_energy = h_subspace.real
        elif i == 1:
            w, v = np.linalg.eigh(h_subspace)
            subspace_energy = w[0]
        else:
            subspace_energy = scipy.sparse.linalg.eigsh(h_subspace, k=1, which="SA")[0][0]


        if subspace_energy < target_energy:
            print("K_overlap = {0:}".format(i + 1))
            K_overlap =  i + 1
            break
    else:
        K_overlap = np.nan
        print("Chemical accuracy not reached, try more vectors per sector")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Optimize orbitals and/or graph structure "
                                                 "to find two-orbital quasisymmetries ('quartets')",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("molpath",
                        help="path to the Hamiltonian (PySCF checkfile)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reference",
                        help="reference state to use in calculations (default: fci)",
                        default="fci")
    parser.add_argument("--optimization_mode",
                        help="Optimization regime. Available keys: \n"
                             "None: No optimization \n"
                             "OO: Orbital optimization for a fixed connection graph",
                        default=None,
                        )
    parser.add_argument("--visualize", action="store_true",
                        help="Draw the NC metrics after optimization")
    parser.add_argument("--quartet_graph", default="ring")
    args = parser.parse_args()

    print("loading the hamiltonian")
    mol = pyscf.lib.chkfile.load_mol(args.molpath)
    mf = pyscf.scf.RHF(mol)
    mf.update_from_chk(args.molpath)
    moldata = ffsim.MolecularData.from_scf(mf)

    print("finding fci") # we need it for energy metrics anyway
    cisolver = pyscf.fci.FCI(mf)
    cisolver.kernel()

    if args.reference == "fci":
        state = np.array(cisolver.ci.reshape((-1,)), dtype="complex") # ffsim will complain without dtype='complex'
    elif args.reference == "hf":
        state = ffsim.hartree_fock_state(moldata.norb, moldata.nelec)
    else:
        raise ValueError("--reference can be 'hf' or 'fci'")

    mo_quartets = all_quartet_commutators(moldata, state, np.eye(moldata.norb))
    mo_quartets_matrix = np.triu(nx.adjacency_matrix(mo_quartets).todense(), 1)
    iu = np.triu_indices(moldata.norb, 1)
    mo_quartets_sorted = np.sort(mo_quartets_matrix[iu])
    sum_of_lowest_mo_quartets = np.sum(mo_quartets_sorted[:moldata.norb])
    print("Sum of lowest m quartets (canonical orbitals)", sum_of_lowest_mo_quartets)

    if args.quartet_graph == "ring":
        quartet_graph = nx.cycle_graph(moldata.norb)
    else:
        raise ValueError()
    print(list(quartet_graph.edges()))

    if args.optimization_mode is None or args.optimization_mode == "None":
        print("No orbital optimization")
        U = np.eye(moldata.norb)
    elif args.optimization_mode == "OO":
        print("Optimizing orbitals keeping the quartet graph fixed")
        f = nc_cost(moldata, state, quartet_graph)
        x0 = np.zeros(comb(moldata.norb, 2))
        print("NC cost, Canonical orbitals", f(x0))
        res = scipy.optimize.minimize(f, x0, method="L-BFGS-B", options={"maxiter": 100})
        print(res)
        U = x_to_rotation(res.x, moldata.norb)
    else:
        raise ValueError()

    optimized_quartets = all_quartet_commutators(moldata, state, U)
    opt_quartets_matrix = np.triu(nx.adjacency_matrix(optimized_quartets).todense(), 1)
    opt_quartets_sorted = np.sort(opt_quartets_matrix[iu])
    sum_of_lowest_opt_quartets = np.sum(opt_quartets_sorted[:moldata.norb])
    print("Sum of lowest m quartets (optimized orbitals)", sum_of_lowest_opt_quartets)

    # with optimized quartets we can partition the function

    best_quartet_indices = np.argsort(opt_quartets_matrix[iu])[:moldata.norb - 1]
    best_quartets = [(iu[0][i], iu[1][i]) for i in best_quartet_indices]
    print("m-1 lowest NC factor quartets")
    print(best_quartets)

    print("creating sectors")
    sectors = quartet_sectors(best_quartets, moldata.norb, moldata.nelec)

    sector_metrics(moldata, state, U, sectors,
                   target_energy=cisolver.e_tot + 0.0016)





    if args.visualize:
        visualize_nc(moldata, state, U)


