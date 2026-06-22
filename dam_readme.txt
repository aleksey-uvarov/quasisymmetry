AI (mistral)-generated explanation with my notes

____________________________________________________________________________________

This code implements a mathematical framework for discovering and exploiting approximate symmetries (quasisymmetries) in molecular electronic Hamiltonians to enable efficient subspace decomposition of the exponentially large Hilbert space.

What Problem This Solves
In quantum many-body physics applied to electronic structure, the Full Configuration Interaction (FCI) method gives the exact solution but scales exponentially with system size. For N electrons in M orbitals, the Hilbert space dimension is C(M, N_α) × C(M, N_β), becoming intractable for even modest molecules.
The standard approach to mitigate this is symmetry adaptation: if the Hamiltonian commutes with symmetry operators (point group symmetries, spin, etc.), the Hilbert space decomposes into invariant subspaces that can be diagonalized separately.
However, for stretched molecules or strongly correlated systems, exact symmetries are broken or absent. This repository develops an alternative: quasisymmetries — operators that approximately commute with H, which still enable efficient subspace decomposition.

Mathematical Framework
Core Objects
The Hamiltonian H is a second-quantized electronic Hamiltonian:
H = Σ h_ij a⁺_i a_j + Σ V_ijkl a⁺_i a⁺_j a_k a_l
Local Parity Operators
For each orbital i (treating α and β separately or combined), define a local parity:
s_i = 1 - 2 n_i      where n_i = a⁺_i a_i
This is a number parity operator: +1 if orbital i is unoccupied, -1 if occupied.
Quasisymmetries
!!! A quasisymmetry is a product of local parities:
S = s_{i₁} s_{i₂} ... s_{i_k}
!!! Both spin-orbital parity s_i,alpha, s_i,beta and orbital parity s_i = s_i,alpha * s_i,beta are implemented. See below
The key insight: while individual s_i do not commute with H, specific products can have small commutators:
||[H, S]|Ψ⟩||² ≈ 0
Here, Ψ is obtained via FCI or HF.
Symmetry Sectors
Each quasisymmetry partitions the Hilbert space into two sectors based on eigenvalue ±1. Multiple quasisymmetries create a direct product decomposition:
ℋ = ⊕_{σ₁,σ₂,...,σ_m} ℋ_{σ₁σ₂...σ_m}
where σ_i ∈ {+1, -1} are the eigenvalues of each quasisymmetry operator.

The Algorithm: Three-Stage Workflow
- Stage 1: Hamiltonian Generation
(make_pyscf_hamiltonian.py)
• Generates molecular Hamiltonians for various systems (H₂, N₂, LiH, H₂O, H₄ in different geometries)
• Uses PySCF to compute one- and two-electron integrals
• Handles both equilibrium and stretched geometries (where exact symmetries vanish)
- Stage 2: Quasisymmetry Discovery
(show_symmetries.py)
• For a given Hamiltonian and reference state (FCI or HF), computes:
NC(i,j) = ||[H, s_i s_j]|Ψ⟩||²
NC(i,i) = ||[H, s_i]|Ψ⟩||²
• Visualizes these as a matrix (log scale)
• Bright spots indicate quasisymmetries with small commutators
• You manually inspect and select which products to use
- Stage 3: Orbital Optimization & Subspace Analysis
(optimize_orbitals.py, metrics.py)
Orbital Optimization:
• Given a binary parity matrix (incidence matrix specifying which s_i enter each quasisymmetry)
• Finds an orbital rotation U that minimizes the total commutator norm:
Cost(U) = Σ_k ||[H(U), S_k]|Ψ⟩||²
• Uses L-BFGS-B optimization over the manifold of unitary rotations
Subspace Decomposition:
• After optimization, constructs symmetry sectors from the quasisymmetry eigenvalues
• Diagonalizes H within each sector separately
• Computes K: the minimal number of states (distributed across sectors) needed to recover the FCI ground state energy to chemical accuracy (1.6 mHa = 0.0016 Hartree)

Why This Matters for Many-Body Physics
Connection to Standard Concepts
• Exact symmetries: In group theory, if [H, S] = 0, S generates a symmetry of H. The Hilbert space decomposes into irreducible representations.
• Quasisymmetries: Here [H, S] ≈ 0 but ≠ 0. The decomposition is approximate but practically useful.
• Subspace expansion: This is a form of selected CI or configuration interaction with selected sectors. The key difference is that selection is based on symmetry structure rather than energy thresholds.

Key Mathematical Results in This Code
 1. The commutator cost function (optimize_orbitals.py:16-32):
• Measures how well a set of quasisymmetries commutes with H for a given state
• Used as the objective for orbital optimization
 2. Sector identification (metrics.py:20-60):
• For each configuration (bitstring), computes its quasisymmetry eigenvalues
• Groups configurations into sectors
 3. Subspace Hamiltonian construction (metrics.py:62-74):
• Builds H restricted to each symmetry sector
 4. K determination (metrics.py:304-342):
• Finds the minimal K such that the lowest K states (across all sectors) span a space containing the exact ground state to chemical accuracy

Physical Systems Investigated
The code studies:
• Diatomics: H₂, N₂, LiH at various bond lengths (including highly stretched)
• Polyatomics: H₂O, linear/square/rectangular H₄
• Basis sets: STO-3G (minimal), cc-pVDZ
• Focus: Systems where exact point group symmetries are broken, but quasisymmetries emerge
The notebooks (e.g., C_FCI vs C_HF.ipynb, HF dependence.ipynb) analyze:
• Comparison of HF vs FCI reference states for quasisymmetry discovery
• The value of K for different molecules and geometries
• The energy error vs. number of states

____________________________________________________________________________________
____________________________________________________________________________________

What quasisymmetries? Short answer: parity of number of elec in clusters of orbitals
of spin orbitals; clusters can have variable number of (spin-)orbitals, and orbital
rotations on top. Long answer (AI):
____________________________________________________________________________________

The Range of Candidate Quasisymmetries
Short answer: All possible products of local parity operators.
Local Parity Operators: The Building Blocks
For each spatial orbital i (or each spin-orbital), define a local parity:
s_i = 1 - 2n_i
where n_i is the number operator for orbital i. Eigenvalues: +1 (unoccupied), -1 (occupied).
There are two variants:
 1. Spatial orbitals: s_i = (1-2n_i^α)(1-2n_i^β) — one operator per spatial orbital
 2. Spin-resolved/spin-orbitals: s_i^α = 1-2n_i^α and s_i^β = 1-2n_i^β — separate operators per spin-orbital

What Gets Tested Automatically
The discovery tool (show_symmetries.py) evaluates and visualizes:
• All first-order candidates: Every individual s_i
• All second-order candidates: Every pairwise product s_i s_j for i < j
It computes for each:
NC = ||[H, S]|Ψ⟩||²
and displays this as a heatmap. Bright spots = good quasisymmetries (small commutator).
This automatic screening only covers spin-averaged, first- and second-order products.

What Can Be Manually Specified
Through the binary parity matrix, you can specify any product of local parities:
• Any order: Single, pairwise, triple, quadruple, ... up to all n orbitals
• Any combination: You pick which local parities to multiply
• Spin-resolved: If using 2×norb columns, you can include s_i^α and s_i^β separately
The full candidate space is: all 2^n - 1 non-trivial products of the local parity operators (where n = norb for spin-averaged, or 2×norb for spin-resolved).

Summary
Automatic test: All s_i and s_i s_j relative to orbitals
Manual specification: Any product of {s_0, s_1, ..., s_{n-1}} or of {s_0^α, s_0^β, s_1^α, s_1^β, ...}
Workflow: use the heatmap to identify promising first/second-order candidates, then manually construct higher-order products if needed.