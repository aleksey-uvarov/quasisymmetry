import numpy as np
import argparse
import openfermion as of
from openfermionpyscf import run_pyscf

from chemistry import  get_geometry_and_description

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol",
                        help="one of the following: lih, h2o, h4_linear, h4_square, h4_rectangle, h2")
    parser.add_argument("bond", type=float, help="bond")
    parser.add_argument("--basis", default="sto-3g")

    args = parser.parse_args()

    geometry, description = get_geometry_and_description(args.mol, args.bond)

    mol = of.MolecularData(
        geometry=geometry,
        basis=args.basis,
        multiplicity=1,
        charge=0,
        description=description
    )
    mol = run_pyscf(mol, run_scf=True, run_fci=False, run_cisd=False)

    mol.filename = "./hamiltonians/" + description

    mol.save()