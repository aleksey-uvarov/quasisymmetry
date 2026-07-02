import numpy as np
import ffsim
import scipy
from scipy.sparse.linalg import LinearOperator
from typing import Callable

def x_to_rotation(x, norb):
    iu = np.triu_indices(norb, k=1)
    rotation_generator = np.zeros((norb, norb))
    rotation_generator[iu] = x
    rotation_generator -= rotation_generator.T
    return scipy.linalg.expm(rotation_generator)

def commutator_cost_v2(moldata: ffsim.MolecularData,
                    symmetries: list,
                    reference_state: np.ndarray) -> Callable:
    def f(x):
        U = x_to_rotation(x, moldata.norb)
        rotated_state = ffsim.apply_orbital_rotation(reference_state,
                                                     U,
                                                     moldata.norb,
                                                     moldata.nelec)
        h = ffsim.linear_operator(moldata.hamiltonian.rotated(U),
                                  norb=moldata.norb, nelec=moldata.nelec)
        total_nc = 0
        h_on_rotate_state = h @ rotated_state
        for s in symmetries:
            term1 = h @ (s @ rotated_state)
            term2 = s @ h_on_rotate_state
            commutator_on_state = term1 - term2
            total_nc += np.linalg.norm(commutator_on_state)**2
        return total_nc
    return f


# Template to design alternative cost functions:

# def build_other_cost_function(moldata: ffsim.MolecularData,
#                    symmetries: list,
#                    reference_state: np.ndarray) -> Callable:
#
# def f(x):
#       # ...
#    return f