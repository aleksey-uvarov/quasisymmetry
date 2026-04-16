import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.linalg import logm, expm
import csv

from openfermion import (
    MolecularData,
    FermionOperator,
    get_fermion_operator,
    jordan_wigner,
    get_sparse_operator,
    normal_ordered,
)
from openfermionpyscf import run_pyscf


@dataclass
class OptLog:
    V: list
    nOmega: list
    x: list

def build_total_operator(U_spatial, n_spatial, a, b, c, tol=1e-12):
    S_ferm = FermionOperator()
    for i in range(n_spatial):
        S_ferm += rotated_seniority_orbital_fermion(
            U_spatial, i, n_spatial, a, b, c, tol=tol
        )
    return normal_ordered(S_ferm)


def evaluate_single_point(molecule: str, x: float, **kwargs):
    t0 = time.time()

    geometry, description = get_geometry_and_description(molecule, x, **kwargs)

    mol = MolecularData(
        geometry=geometry,
        basis=BASIS,
        multiplicity=MULTIPLICITY,
        charge=CHARGE,
        description=description,
    )
    mol = run_pyscf(mol, run_scf=True, run_fci=True, run_cisd=True)

    n_e = mol.n_electrons
    n_spatial = mol.n_orbitals
    n_qubits = 2 * n_spatial
    dim = 1 << n_qubits

    # Hamiltonian
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

    # CISD reference
    hf_b = closed_shell_hf_bitstring(n_e, n_spatial)
    # E_cisd, _, _ = solve_cisd_state(H_sub, basis_bitstrings, hf_b, n_qubits)

    # RDMs from FCI
    gamma_a, gamma_b, Gamma_ab = compute_spin_rdms_from_statevector(psi_full, n_spatial)

    # Identity / baseline coefficients
    pairs = pair_list_for_n(n_spatial)
    m = len(pairs)

    x_id = np.zeros(m + 2)
    x_id[m] = np.arccos(-2.0 / np.sqrt(6.0))   # c = -2/sqrt(6)
    x_id[m + 1] = np.pi / 4.0                  # a = b = 1/sqrt(6)

    V_id, _, U_id, a_id, b_id, c_id = variance_restricted(
        gamma_a, gamma_b, Gamma_ab, x_id, pairs
    )

    # Optimized coefficients/orbitals
    best = optimize_variance_restricted(gamma_a, gamma_b, Gamma_ab)
    res = best["res"]
    V_opt, _, U_opt, a_opt, b_opt, c_opt = variance_restricted(
        gamma_a, gamma_b, Gamma_ab, res.x, best["pairs"]
    )

    # State-specific commutativity on FCI state
    if EVAL_STATE_SPECIFIC_COMMUTATIVITY:
        S_id_ferm = build_total_operator(U_id, n_spatial, a_id, b_id, c_id, tol=OP_COEF_TOL)
        S_opt_ferm = build_total_operator(U_opt, n_spatial, a_opt, b_opt, c_opt, tol=OP_COEF_TOL)

        S_id_mat = fermion_to_sparse_qubit(S_id_ferm, n_qubits)
        S_opt_mat = fermion_to_sparse_qubit(S_opt_ferm, n_qubits)

        nc_id = comm_state_norm(H_full, S_id_mat, psi_full, check_eigenstate=True)
        nc_opt = comm_state_norm(H_full, S_opt_mat, psi_full, check_eigenstate=True)
    else:
        nc_id = 0.0
        nc_opt = 0.0
    # ------------------------------------------------------------
    # Symmetry-leakage check on the FCI state:
    # S|psi> = s|psi> + |delta>
    # ------------------------------------------------------------
    leak_id = analyze_symmetry_leakage(
        H_full, S_id_mat, psi_full, label=f"{molecule} / identity"
    )
    leak_opt = analyze_symmetry_leakage(
        H_full, S_opt_mat, psi_full, label=f"{molecule} / optimized"
    )

    # Entropy in the rotated determinant basis
    H_id = H_sub.toarray().astype(np.complex128)
    psi_id = v_sub / np.linalg.norm(v_sub)

    sectors_id = build_generalized_sectors(
        basis_bitstrings, n_spatial, n_qubits, a_id, b_id, c_id
    )
    I_S_id, I_SS_id, _ = shannon_block_decomposition(H_id, psi_id, sectors_id)

    sectors_opt = build_generalized_sectors(
        basis_bitstrings, n_spatial, n_qubits, a_opt, b_opt, c_opt
    )
    R_opt = orbital_rotation_representation_R(U_opt, basis_bitstrings, n_spatial)
    H_U = R_opt.conj().T @ (H_id @ R_opt)
    H_U = 0.5 * (H_U + H_U.conj().T)
    psi_U = R_opt.conj().T @ psi_id
    I_S_opt, I_SS_opt, _ = shannon_block_decomposition(H_U, psi_U, sectors_opt)

    elapsed = time.time() - t0
    print(
        f"[{molecule}] x={x:.4f} finished in {elapsed:.2f} s | "
        f"V_opt={V_opt:.6f} | (a,b,c)=({a_opt:.4f},{b_opt:.4f},{c_opt:.4f})"
    )

    return {
        "Molecule": molecule,
        "Geometry_Param": x,
        "E_FCI": E_fci,
        # "E_CISD": E_cisd,
        "V_Identity": V_id,
        "V_Optimized": V_opt,
        "a": a_opt,
        "b": b_opt,
        "c": c_opt,
        "NC_Identity": nc_id,
        "NC_Optimized": nc_opt,
        "Coarse_Entropy_Identity": I_SS_id,
        "Coarse_Entropy_Optimized": I_SS_opt,
        "Fine_Entropy_Identity": I_S_id,
        "Fine_Entropy_Optimized": I_S_opt,
    }


def variance_restricted(gamma_a, gamma_b, Gamma_ab, x_params, pairs):
    n = gamma_a.shape[0]
    m = len(pairs)

    # Unpack orbital rotations and operator angles
    thetas = x_params[:m]
    phi1, phi2 = x_params[m], x_params[m+1]

    # Spherical parameterization for sqrt(a^2 + b^2 + c^2) = 1
    a = np.sin(phi1) * np.cos(phi2)
    b = np.sin(phi1) * np.sin(phi2)
    c = np.cos(phi1)

    U = build_U_from_thetas(n, thetas, pairs)
    Ua = U.T @ gamma_a @ U
    Ub = U.T @ gamma_b @ U

    exp_vals = np.zeros(n, dtype=float)
    V_total = 0.0

    for i in range(n):
        u = U[:, i]
        G_i = np.einsum("p,q,r,s,pqrs->", u, u, u, u, Gamma_ab, optimize=True).real
        N_a = Ua[i, i].real
        N_b = Ub[i, i].real

        # < \tilde{\Omega}_i >
        exp_omega = a * N_a + b * N_b + c * G_i
        exp_vals[i] = exp_omega

        # < \tilde{\Omega}_i^2 >
        exp_omega_sq = a**2 * N_a + b**2 * N_b + (2*a*b + 2*a*c + 2*b*c + c**2) * G_i

        # Exact Variance: <O^2> - <O>^2
        V_total += float(exp_omega_sq - exp_omega**2)

    return V_total, exp_vals, U, a, b, c


def get_geometry_and_description(molecule: str, x: float, **kwargs):
    mol = molecule.lower()

    if mol == "lih":
        return build_lih_geometry(x), f"LiH_Bond{x:.4f}"

    # elif mol == "h2o":
    #     angle = kwargs.get("hoh_angle_deg", 104.5)
    #     return build_h2o_geometry(x, hoh_angle_deg=angle), f"H2O_OH{x:.4f}"
    #
    # elif mol == "h4_linear":
    #     return build_h4_linear_geometry(x), f"H4_linear_d{x:.4f}"
    #
    # if mol == "h4_square":
    #     return build_h4_square_geometry(x), f"H4_square_side{x:.4f}"
    #
    # elif mol == "h4_rectangle":
    #     aspect_ratio = kwargs.get("aspect_ratio", 1.5)
    #     return build_h4_rectangle_geometry(x, aspect_ratio=aspect_ratio), f"H4_rectangle_long{x:.4f}_ar{aspect_ratio:.3f}"

    else:
        raise ValueError(
            f"Unsupported molecule '{molecule}'. "
            "Choose from: lih, h2o, h4_linear, h4_square, h4_rectangle"
        )


def build_lih_geometry(li_h_bond_angstrom: float):
    r = li_h_bond_angstrom / 2.0
    return [
        ("Li", (0.0, 0.0, -r)),
        ("H",  (0.0, 0.0, +r)),
    ]


def popcount(x: int) -> int:
    return int(x.bit_count())


def closed_shell_hf_bitstring(n_electrons, n_spatial):
    if n_electrons % 2 != 0:
        raise ValueError("This helper assumes closed-shell (even electron count).")

    n_qubits = 2 * n_spatial
    occ = n_electrons // 2
    b = 0

    for i in range(occ):
        a_mode = 2 * i
        b_mode = 2 * i + 1
        b |= (1 << mode_to_bitpos(a_mode, n_qubits))
        b |= (1 << mode_to_bitpos(b_mode, n_qubits))

    return b


def mode_to_bitpos(mode: int, n_qubits: int) -> int:
    """
    OpenFermion-consistent mapping inferred from your identity check:
    fermionic mode 0 is the LEFTMOST bit in the printed binary string.
    """
    if not (0 <= mode < n_qubits):
        raise ValueError(f"mode {mode} out of range for n_qubits={n_qubits}")
    return n_qubits - 1 - mode


def compute_spin_rdms_from_statevector(statevec, n_spatial):
    n_qubits = 2 * n_spatial
    dim = 1 << n_qubits
    if statevec.shape[0] != dim:
        raise ValueError("state dim doesn't match")

    psi = statevec
    gamma_a = np.zeros((n_spatial, n_spatial), dtype=np.complex128)
    gamma_b = np.zeros((n_spatial, n_spatial), dtype=np.complex128)
    Gamma_ab = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial), dtype=np.complex128)

    nz = np.nonzero(np.abs(psi) > 0)[0]

    def fill_gamma(gamma, spin_offset):
        for q in range(n_spatial):
            q_mode = 2 * q + spin_offset
            for x in nz:
                amp_x = psi[x]
                x1, s1 = apply_annihilate(int(x), q_mode, n_qubits)
                if x1 is None:
                    continue
                for p in range(n_spatial):
                    p_mode = 2 * p + spin_offset
                    x2, s2 = apply_create(x1, p_mode, n_qubits)
                    if x2 is None:
                        continue
                    gamma[p, q] += np.conjugate(psi[x2]) * amp_x * (s1 * s2)

    fill_gamma(gamma_a, 0)
    fill_gamma(gamma_b, 1)

    for p in range(n_spatial):
        p_mode = 2 * p
        for q in range(n_spatial):
            q_mode = 2 * q + 1
            for r in range(n_spatial):
                r_mode = 2 * r
                for s in range(n_spatial):
                    s_mode = 2 * s + 1
                    val = 0.0 + 0.0j
                    for x in nz:
                        amp_x = psi[x]
                        x1, sr = apply_annihilate(int(x), r_mode, n_qubits)
                        if x1 is None:
                            continue
                        x2, ss = apply_annihilate(x1, s_mode, n_qubits)
                        if x2 is None:
                            continue
                        x3, sq = apply_create(x2, q_mode, n_qubits)
                        if x3 is None:
                            continue
                        x4, sp_ = apply_create(x3, p_mode, n_qubits)
                        if x4 is None:
                            continue
                        val += np.conjugate(psi[x4]) * amp_x * (sr * ss * sq * sp_)
                    Gamma_ab[p, q, r, s] = val

    return gamma_a, gamma_b, Gamma_ab


def apply_annihilate(bitstring: int, mode: int, n_qubits: int):
    pos = mode_to_bitpos(mode, n_qubits)
    if ((bitstring >> pos) & 1) == 0:
        return None, 0
    sign = parity_sign(bitstring, mode, n_qubits)
    return bitstring & ~(1 << pos), sign


def apply_create(bitstring: int, mode: int, n_qubits: int):
    pos = mode_to_bitpos(mode, n_qubits)
    if ((bitstring >> pos) & 1) == 1:
        return None, 0
    sign = parity_sign(bitstring, mode, n_qubits)
    return bitstring | (1 << pos), sign


def parity_sign(bitstring: int, mode: int, n_qubits: int) -> int:
    """
    Fermionic JW sign for acting on 'mode':
    (-1)^(number of occupied modes with label < mode).
    IMPORTANT: this is NOT the same as counting lower integer bit positions
    once mode 0 is mapped to the MSB.
    """
    occ_before = 0
    for k in range(mode):
        occ_before += mode_is_occupied(bitstring, k, n_qubits)
    return -1 if (occ_before % 2 == 1) else 1


def mode_is_occupied(bitstring: int, mode: int, n_qubits: int) -> int:
    pos = mode_to_bitpos(mode, n_qubits)
    return (bitstring >> pos) & 1


def pair_list_for_n(n): #order of givens
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def build_U_from_thetas(n, thetas, pairs): #U size = n_spatial x n_spatial
    U = np.eye(n)
    for th, (p, q) in zip(thetas, pairs):
        U = U @ givens(n, p, q, th)
    return U


#copy def from wikipedia
def givens(n, p, q, theta):
    G = np.eye(n)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    G[p, p] = c
    G[q, q] = c
    G[p, q] = s
    G[q, p] = -s
    return G


def optimize_variance_restricted(gamma_a, gamma_b, Gamma_ab):
    np.random.seed(RANDOM_SEED)
    n = gamma_a.shape[0]
    pairs = pair_list_for_n(n)
    m = len(pairs)
    num_params = m + 2 # +2 for phi1, phi2

    def obj(x):
        V, _, _, _, _, _ = variance_restricted(gamma_a, gamma_b, Gamma_ab, x, pairs)
        return V

    best = None
    for r in range(N_RESTARTS):
        x0 = np.zeros(num_params)
        if r == 0:
            # Initialize close to standard seniority: a=1, b=1, c=-2 (Normalized by sqrt(6))
            x0[m] = np.arccos(-2.0 / np.sqrt(6.0)) # phi1 for c
            x0[m+1] = np.pi / 4.0                  # phi2 for a, b
        else:
            x0[:m] = ANGLE_INIT_SCALE * np.random.randn(m)
            x0[m] = np.random.uniform(0, np.pi)
            x0[m+1] = np.random.uniform(0, 2*np.pi)

        log = OptLog(V=[], nOmega=[], x=[])

        def callback(xk):
            V, nO, _, _, _, _ = variance_restricted(gamma_a, gamma_b, Gamma_ab, xk, pairs)
            log.V.append(V); log.nOmega.append(nO); log.x.append(np.array(xk, copy=True))

        if OPT_METHOD.upper() == "POWELL":
            res = minimize(obj, x0=x0, method="Powell", options={"maxiter": MAXITER, "disp": False})
            callback(res.x)
        else:
            res = minimize(obj, x0=x0, method=OPT_METHOD, options={"maxiter": MAXITER, "disp": False}, callback=callback)

        V_fin = obj(res.x)
        if best is None or V_fin < best["V"]:
            best = {"res": res, "log": log, "V": V_fin, "pairs": pairs, "a": np.sin(res.x[m])*np.cos(res.x[m+1]), "b": np.sin(res.x[m])*np.sin(res.x[m+1]), "c": np.cos(res.x[m])}

    return best

def fermion_to_sparse_qubit(op_fermion, n_qubits): # in qubit matrix
    op_qubit = jordan_wigner(op_fermion)
    return get_sparse_operator(op_qubit, n_qubits).tocsc()


def rotated_seniority_orbital_fermion(U_spatial, i_spatial, n_spatial, a, b, c, tol=1e-12):
    n_a = rotated_number_operator_fermion(U_spatial, i_spatial, spin_offset=0, n_spatial=n_spatial, tol=tol)
    n_b = rotated_number_operator_fermion(U_spatial, i_spatial, spin_offset=1, n_spatial=n_spatial, tol=tol)

    # Generalized operator
    omega = normal_ordered(a * n_a + b * n_b + c * (n_a * n_b))
    return omega


def rotated_number_operator_fermion(U_spatial, i_spatial, spin_offset, n_spatial, tol=1e-12):
    op = FermionOperator()
    for p in range(n_spatial):
        for q in range(n_spatial):
            coef = np.conjugate(U_spatial[p, i_spatial]) * U_spatial[q, i_spatial]
            if abs(coef) <= tol:
                continue
            p_mode = 2 * p + spin_offset
            q_mode = 2 * q + spin_offset
            op += FermionOperator(((p_mode, 1), (q_mode, 0)), coef)
    return op


def comm_state_norm(H_mat, S_mat, psi, check_eigenstate=False, atol=1e-10):
    """
    Returns || [H,S] psi ||.
    Optionally verifies the eigenstate identity if psi is an eigenstate of H.
    """
    Apsi = H_mat.dot(S_mat.dot(psi)) - S_mat.dot(H_mat.dot(psi))
    norm2 = np.vdot(Apsi, Apsi)

    # Optional consistency check:
    # <psi|[H,S]^2|psi> = - <Apsi|Apsi>
    A2psi = H_mat.dot(S_mat.dot(Apsi)) - S_mat.dot(H_mat.dot(Apsi))
    exp = np.vdot(psi, A2psi)
    if not np.allclose(exp, -norm2, atol=atol):
        raise AssertionError("Expected <psi|[H,S]^2|psi> = -<Apsi|Apsi>")

    if check_eigenstate:
        E0 = np.vdot(psi, H_mat.dot(psi))
        resid = np.linalg.norm(H_mat.dot(psi) - E0 * psi)
        if resid < atol:
            Spsi = S_mat.dot(psi)
            HSpsi = H_mat.dot(Spsi)
            norm3 = (
                np.vdot(HSpsi, HSpsi)
                - 2 * E0 * np.vdot(Spsi, HSpsi)
                + E0**2 * np.vdot(Spsi, Spsi)
            )
            if not np.allclose(norm2, norm3, atol=atol):
                raise AssertionError("Eigenstate expansion check failed.")

    return float(np.sqrt(np.real(norm2)))


def analyze_symmetry_leakage(H_mat, S_mat, psi, label="", atol=1e-12):
    """
    For normalized psi, decompose
        S|psi> = s|psi> + |delta>
    where s = <psi|S|psi> and <psi|delta> = 0.

    Prints:
      - <psi|S|psi>
      - ||delta||
      - <delta|H|delta>
      - normalized delta energy = <delta|H|delta> / <delta|delta>
      - excitation above E0 if psi is an eigenstate of H
    """
    psi = np.asarray(psi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)

    Spsi = S_mat.dot(psi)
    s = np.vdot(psi, Spsi)
    delta = Spsi - s * psi

    delta_norm2 = np.real(np.vdot(delta, delta))
    delta_norm = float(np.sqrt(max(delta_norm2, 0.0)))

    E0 = np.real(np.real(np.vdot(psi, H_mat.dot(psi))))
    delta_H_delta = np.real(np.vdot(delta, H_mat.dot(delta)))

    print(f"\n[{label}] symmetry-leakage check")
    print(f"  <psi|S|psi>                = {s.real:+.12f}{s.imag:+.3e}j")
    print(f"  ||delta||                  = {delta_norm:.12e}")
    print(f"  <delta|H|delta>            = {delta_H_delta.real:+.12e}{delta_H_delta.imag:+.3e}j")

    if delta_norm2 > atol:
        E_delta = delta_H_delta / delta_norm2
        print(f"  <H>_delta = <delta|H|delta>/<delta|delta> = {E_delta.real:+.12f}{E_delta.imag:+.3e}j")
        print(f"  <H>_delta - E0             = {E_delta.real - E0:+.12e}")
    else:
        print("  <H>_delta                  = undefined (delta ~ 0)")

    return {
        "s": s,
        "delta": delta,
        "delta_norm": delta_norm,
        "delta_norm2": delta_norm2,
        "delta_H_delta": delta_H_delta,
        "E0": E0,
    }


def build_generalized_sectors(basis_bitstrings, n_spatial, n_qubits, a, b, c, tol_decimals=8):
    """
    Partitions the Hilbert space into sectors defined by the unique
    eigenvalues of the a,b,c parameterized operator.
    """
    sectors = {}
    for k, bit_str in enumerate(basis_bitstrings):
        val = generalized_eigenvalue_from_bitstring(int(bit_str), n_spatial, n_qubits, a, b, c)

        # Round the float to avoid precision mismatches creating artificial sectors
        val_key = round(val, tol_decimals)
        sectors.setdefault(val_key, []).append(k)

    return sectors


def generalized_eigenvalue_from_bitstring(bitstring: int, n_spatial: int, n_qubits: int, a: float, b: float, c: float) -> float:
    """
    Calculates the exact scalar eigenvalue of the generalized
    quasi-symmetry operator for a given determinant bitstring.
    """
    eigval = 0.0
    for i in range(n_spatial):
        oa = mode_is_occupied(bitstring, 2 * i, n_qubits)
        ob = mode_is_occupied(bitstring, 2 * i + 1, n_qubits)
        eigval += (a * oa) + (b * ob) + (c * (oa * ob))
    return eigval


def shannon_block_decomposition(H_dense, psi_vec, sectors_dict):
    weights_fine = []
    sector_weights = {}

    for msk, idxs in sectors_dict.items():
        psi_s = psi_vec[idxs]
        ws = float(np.vdot(psi_s, psi_s).real)
        sector_weights[msk] = ws

        d = len(idxs)
        if d == 0:
            continue

        H_blk = H_dense[np.ix_(idxs, idxs)]
        H_blk = 0.5 * (H_blk + H_blk.conj().T)

        evals_blk, evecs_blk = np.linalg.eigh(H_blk)
        c_eig = evecs_blk.conj().T @ psi_s
        weights_fine.extend((np.abs(c_eig) ** 2).tolist())

    I_S = shannon_entropy_from_weights(weights_fine)
    I_SS = shannon_entropy_from_weights(list(sector_weights.values()))

    p_sum = float(np.sum(weights_fine))
    if abs(p_sum - 1.0) > 1e-6:
        print(f"  [warn] Σ_{'{'}s,i{'}'} w_s,i = {p_sum:.8f} (expected ~1).")

    return I_S, I_SS, sector_weights


def shannon_entropy_from_weights(weights, eps=1e-15):
    w = np.asarray(weights, dtype=float)
    w = w[w > eps]
    return float(-np.sum(w * np.log(w))) if w.size else 0.0


def orbital_rotation_representation_R(U_spatial, basis_bitstrings, n_spatial, tol=1e-12):
    n_qubits = 2 * n_spatial
    idx = np.asarray(basis_bitstrings, dtype=int)

    U_spatial = np.asarray(U_spatial, dtype=np.complex128)
    K = logm(U_spatial)
    K = 0.5 * (K - K.conj().T)  # enforce anti-Hermitian numerically

    # 2) Lift K to the fermionic generator κ = sum_{pqσ} K_pq a†_{pσ} a_{qσ}
    kappa = FermionOperator()
    for p in range(n_spatial):
        for q in range(n_spatial):
            coef = K[p, q]
            if abs(coef) <= tol:
                continue
            # alpha
            p_a = 2 * p
            q_a = 2 * q
            kappa += FermionOperator(((p_a, 1), (q_a, 0)), coef)
            # beta
            p_b = 2 * p + 1
            q_b = 2 * q + 1
            kappa += FermionOperator(((p_b, 1), (q_b, 0)), coef)

    # 3) Matrix of κ on the full Fock space, then restrict to the fixed-N subspace
    kappa_mat_full = fermion_to_sparse_qubit(kappa, n_qubits)
    kappa_sub = kappa_mat_full[idx, :][:, idx].toarray().astype(np.complex128)

    # 4) Exponentiate on the fixed-N subspace
    R_sub = expm(kappa_sub)

    return R_sub


if __name__=="__main__":

    ### The molecule stuff should be imported from an fcidump or something,
    # you shouldn't recreate Hamiltonians on the spot every time

    LIH_BOND_ANGSTROM = 1.60  # ~equilibrium Li–H bond length rough starting point (can vary by basis/method)
    # HOH_ANGLE_DEG    = 104.5
    CHARGE = 0
    MULTIPLICITY = 1
    BASIS = "sto-3g"

    ####
    OPT_METHOD = "Powell"
    MAXITER = 200
    N_RESTARTS = 5
    ANGLE_INIT_SCALE = 0.2
    RANDOM_SEED = 0
    TOPK_ANGLES_TO_PRINT = 10

    # for state spec commuta
    EVAL_STATE_SPECIFIC_COMMUTATIVITY = True
    OP_COEF_TOL = 1e-12

    molecule = "lih"

    evaluate_single_point(molecule, LIH_BOND_ANGSTROM)