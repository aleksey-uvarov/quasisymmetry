import argparse
import numpy as np
import time
import ffsim
import scipy
import pyscf
import pyscf.fci

from typing import Callable
from math import comb
from functools import cache, reduce

# from chemistry import load_moldata, fcidump_data # may need to modify path


#TODO implement following two funs, look at optimize_orbitals.parities, also implement tests
def build_one_orb_num_operators(norb, nelec):
    return 0

#TODO
def build_two_orb_num_operators(norb, nelec):
    return 0

#TODO implement following optimize_orbitals.parity_matrix_to_quasisymmetries
def cluster_matrix_to_cluster_number_quasisymmetries(cluster_matrix: np.ndarray,
                                     norb,
                                     nelec):
    return 0

#TODO implement based on metrics.symmetry_sectors
def cluster_number_symmetry_sectors(cluster_matrix, norb, nelec):
    return 0