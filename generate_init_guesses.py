import numpy as np
import argparse

from itertools import combinations

from chemistry import get_mol

SENIORITY_ANGLES = (np.arccos(-2.0 / np.sqrt(6.0)), np.pi / 4.0)

OR_POP_ANGLES = (np.arccos(-1 / np.sqrt(3.0)), np.pi / 4.0) # a = b = -c

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol",
        help="one of the following: lih, h2o, h4_linear, h4_square, h4_rectangle, h2")
    parser.add_argument("bond", type=float, help="bond")
    parser.add_argument("--npoints", default=3, type=int)
    parser.add_argument("--mode", default="perturb_U")
    parser.add_argument("--noise_scale", default=-6, type=int)

    args = parser.parse_args()
    mol = get_mol(args.mol, args.bond)

    pairs = list(combinations(range(mol.n_orbitals), 2))
    m = len(pairs)

    xs = np.zeros((args.npoints, m + 2))

    rng = np.random.default_rng()

    if args.mode == "perturb_U":
        for i in range(args.npoints):
            xs[i, :m] = rng.normal(scale=10**(args.noise_scale), size=m)
            xs[i, m:] = SENIORITY_ANGLES

    np.savetxt("x0_" + args.mol + "_" + str(args.bond)
               + "_" + args.mode + "_" + str(args.noise_scale) + ".txt", xs)