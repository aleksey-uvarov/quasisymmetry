import numpy as np
import openfermion as of
from openfermionpyscf import run_pyscf


d = 2.067
coords = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]
geometry = [("H", (0.0, 0.0, z)) for z in coords]

mol = of.MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0
    )
mol = run_pyscf(mol, run_scf=True, run_fci=False, run_cisd=False)

print(np.linalg.det(mol.canonical_orbitals))