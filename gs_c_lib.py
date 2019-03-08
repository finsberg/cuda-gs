import os
import ctypes
from ctypes import c_int, c_long, c_ulong, c_double
import subprocess

import numpy as np

float64_array_2d = np.ctypeslib.ndpointer(dtype=c_double, ndim=2,
                                          flags="contiguous")

def _load_libgs(rebuild=True):
    lib_filename = "libgs.so"
    libdir = os.path.dirname(__file__)
    if rebuild:
        args = ['make', '-C', libdir, lib_filename]
        cp = subprocess.run(
            args,
            capture_output=True,
        )
        if cp.returncode != 0:
            print(cp.stdout.decode('utf-8'))
            print(cp.stderr.decode('utf-8'))
            raise RuntimeError("Failed to build C library")
    lib_path = os.path.join(libdir, lib_filename)
    libgs = np.ctypeslib.load_library(lib_path, '.')
    _setup_functions(libgs)
    return libgs

def _setup_functions(libgs):
    _setup_function_gs_orthogonalise_vector(libgs.gs_orthogonalise_vector)
    _setup_function_gs_orthogonalise_vector(libgs.gs_orthogonalise_vector_omp)

def _setup_function_gs_orthogonalise_vector(c_func):
    c_func.restype = None # void
    c_func.argtypes = [
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

def orthogonalise_vector_omp(V, new_vec_ind):
    assert len(V.shape) == 2
    M, N = V.shape
    if M >= N:
        print ("Warning: M >= N, {0} >= {1}.".format(M, N))
    _libgs.gs_orthogonalise_vector_omp(V, M, N, new_vec_ind)
