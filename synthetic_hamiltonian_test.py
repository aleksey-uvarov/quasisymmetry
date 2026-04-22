from optimize_for_commutator import *
from itertools import product
import openfermion as of

G_1212 = 0
G_1112 = 1


def coeff_class(i, j, k, el):
    modified_coeffs = [i, j, k, el]
    if i > j:
        modified_coeffs[0], modified_coeffs[1] = modified_coeffs[1], modified_coeffs[0]
    if k > el:
        modified_coeffs[2], modified_coeffs[3] = modified_coeffs[3], modified_coeffs[2]
    if (10 * i + j) > (10 * k + el):
        modified_coeffs[0], modified_coeffs[2] = modified_coeffs[2], modified_coeffs[0]
        modified_coeffs[1], modified_coeffs[3] = modified_coeffs[3], modified_coeffs[1]
    modified_coeffs = tuple(modified_coeffs)
    if modified_coeffs == (0, 0, 0, 0):
        return 0
    elif modified_coeffs == (0, 0, 0, 1):
        return G_1112
    elif modified_coeffs == (0, 0, 1, 1):
        return 0
    elif modified_coeffs == (0, 1, 0, 1):
        return G_1212
    elif modified_coeffs == (0, 1, 1, 1):
        return 0
    elif modified_coeffs == (1, 1, 1, 1):
        return 0
    else:
        print(modified_coeffs)
        raise ValueError()


def vanishing_commutator_model_hamiltonian():
    two_body_terms = [((2 * i + sigma, 1), (2 * k + tau, 1), (2 * el + tau, 0), (2 * j + sigma, 0))
                      for i, j, k, el, sigma, tau in product(range(2), repeat=6)]

    two_body_coeffs = [coeff_class(i, j, k, el)
                       for i, j, k, el, sigma, tau in product(range(2), repeat=6)]

    two_body_hamiltonian = of.FermionOperator()
    for i, term in enumerate(two_body_terms):
        two_body_hamiltonian += of.FermionOperator(term, two_body_coeffs[i] / 2)
    return two_body_hamiltonian

if __name__=="__main__":
    n_orbitals = 2
    n_qubits = 4
    n_e = 2

    dim = 1 << n_qubits

    H_ferm = vanishing_commutator_model_hamiltonian()
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

    pairs = list(combinations(range(n_orbitals), 2))
    m = len(pairs)

    gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_statevector(
        psi_full, n_orbitals)

    print("a, b, c are the same for all orbitals")
    print(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

    x_id = np.zeros(m + 2)
    # x_id[m] = np.arccos(-2.0 / np.sqrt(6.0))  # c = -2/sqrt(6)
    # x_id[m + 1] = np.pi / 4.0  # a = b = 1/sqrt(6)

    x_id[m] = np.arccos(-1.0 / np.sqrt(3.))  # c = -1/sqrt(3)
    x_id[m + 1] = np.pi / 4.0  # a = b = 1/sqrt(3)

    def f(x):
        a = np.sin(x[m]) * np.cos(x[m + 1])
        b = np.sin(x[m]) * np.sin(x[m + 1])
        c = np.cos(x[m])
        # a = 1
        # b = 1
        # c = -1
        U = build_U_from_thetas(n_orbitals, x[:m], pairs)
        total_commutator_norm = 0
        for i in range(n_orbitals):
            # Si_ferm = build_single_local_operator(U, mol.n_orbitals, i,
            #                                       [(a, b, c)] * mol.n_orbitals)
            Si_ferm = normal_ordered(
                rotated_seniority_orbital_fermion(
                    U, i, n_orbitals, a, b, c
                )
            )
            Si_mat = fermion_to_sparse_qubit(Si_ferm, n_qubits)
            total_commutator_norm += expected_squared_commutator(
                H_full, Si_mat, psi_full)
        return total_commutator_norm


    rng = np.random.default_rng()
    print("||[H, S] psi||^2 before optimization", f(x_id))

    variance_before, _, _, _, _, _ = variance_restricted(
        gamma_a, gamma_b, gamma_ab, x_id, pairs
    )
    print("Var S before optimization", variance_before)

    print()

    best_cost = np.inf
    res = None

    for j in range(2):

        dx = rng.normal(scale=1e-6, size=x_id.shape[0])
        dx[-2] = rng.uniform(0, np.pi)
        dx[-1] = rng.uniform(0, 2 * np.pi)

        print("||[H, S] psi||^2 perturbed x0", f(x_id + dx))

        res_current = minimize(f, x_id + dx,
                               method="L-BFGS-B",
                               # method="Powell",
                               options={"maxiter": 100})

        if res_current.fun < best_cost:
            best_cost = res_current.fun
            res = res_current

        print("||[H, S] psi||^2 after optimization", res_current.fun)

        phi1, phi2 = res_current.x[m], res_current.x[m + 1]

        # Spherical parameterization for sqrt(a^2 + b^2 + c^2) = 1
        a_opt = np.sin(phi1) * np.cos(phi2)
        b_opt = np.sin(phi1) * np.sin(phi2)
        c_opt = np.cos(phi1)

        print("Optimal abc (rescaled):", 1, b_opt / a_opt, c_opt / a_opt)

        print(res_current.message)

    print(res.x)

    phi1, phi2 = res.x[m], res.x[m + 1]

    # Spherical parameterization for sqrt(a^2 + b^2 + c^2) = 1
    a_opt = np.sin(phi1) * np.cos(phi2)
    b_opt = np.sin(phi1) * np.sin(phi2)
    c_opt = np.cos(phi1)

    print("Optimal abc (rescaled):", 1, b_opt / a_opt, c_opt / a_opt)

    variance_after, _, _, _, _, _ = variance_restricted(
        gamma_a, gamma_b, gamma_ab, res.x, pairs
    )

    print("Var S after optimization", variance_after)

