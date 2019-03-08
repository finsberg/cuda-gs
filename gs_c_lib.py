import os
import ctypes
from ctypes import c_int, c_long, c_ulong, c_double

import numpy as np

float64_array_2d = np.ctypeslib.ndpointer(dtype=c_double, ndim=2,
                                          flags="contiguous")

def _load_libgs():
    lib_filename = "libgs.so"
    lib_path = os.path.join(os.path.dirname(__file__), lib_filename)
    libgs = np.ctypeslib.load_library(lib_path, '.')
    _setup_functions(libgs)
    return libgs

def _setup_functions(libgs):
    _setup_function_gs_orthogonalise_vector(libgs)

def _setup_function_gs_orthogonalise_vector(libgs):
    libgs.gs_orthogonalise_vector.restype = None # void
    libgs.gs_orthogonalise_vector.argtypes = [
        float64_array_2d, # V
        c_int,            # M
        c_int,            # N
        c_int,            # new_vec_ind
    ]


_libgs = _load_libgs()

def orthogonalise_vector(V, new_vec_ind):
    assert len(V.shape) == 2
    M, N = V.shape
    if M >= N:
        print ("Warning: M >= N, {0} >= {1}.".format(M, N))
    _libgs.gs_orthogonalise_vector(V, M, N, new_vec_ind)
