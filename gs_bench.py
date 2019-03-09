import time
import numba
import numpy as np

has_numba = False
try:
    import numba
    has_numba = True
except ImportError:
    pass

has_cupy = False
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    pass

import gs_c_lib

has_gs_cuda_lib = False
try:
    import gs_cuda_lib
    has_gs_cuda_lib = True
except ImportError:
    pass


def gs_numpy(V):
    for j in range(1,N):
        inner_prods_uv = np.sum(V[:,j,None]*V[:,:j], axis=0)
        inner_prods_uu= np.sum(V[:,:j]*V[:,:j],axis=0)
        V[:,j] -= np.sum(inner_prods_uv/inner_prods_uu*V[:,:j],axis=1)

def gs_numpy_T(V):
    for j in range(1,N):
        inner_prods_uv = np.sum(V[j,:]*V[:j,:], axis=1)
        inner_prods_uu= np.sum(V[:j,:]*V[:j,:], axis=1)
        V[j,:] -= np.sum((inner_prods_uv/inner_prods_uu)[:,np.newaxis]*V[:j,:],axis=0)

@numba.jit(cache=True, nopython=True)
def gs_numba(V):
    for j in range(1,N):
        for i in range(j):
            inner_prod_uv = np.sum(V[:,j]*V[:,i], axis=0)
            inner_prod_uu= np.sum(V[:,i]*V[:,i],axis=0)
            V[:,j] -= inner_prod_uv/inner_prod_uu*V[:,i]

@numba.jit(cache=True, nopython=True)
def gs_numba_T(V):
    for j in range(1,N):
        for i in range(j):
            inner_prod_uv = np.dot(V[j,:],V[i,:])
            inner_prod_uu= np.dot(V[i,:],V[i,:])
            V[j,:] -= (inner_prod_uv/inner_prod_uu)*V[i,:]

def gs_cupy(V):
    for j in range(1,N):
        inner_prods_uv = cp.sum(V[:,j,None]*V[:,:j], axis=0)
        inner_prods_uu= cp.sum(V[:,:j]*V[:,:j],axis=0)
        V[:,j] -= cp.sum(inner_prods_uv/inner_prods_uu*V[:,:j],axis=1)

def gs_cupy_T(V):
    for j in range(1,N):
        inner_prods_uv = cp.sum(V[j,:]*V[:j,:], axis=1)
        inner_prods_uu = cp.sum(V[:j,:]*V[:j,:],axis=1)
        V[j,:] -= cp.sum((inner_prods_uv/inner_prods_uu)[:,np.newaxis]*V[:j,:],axis=0)

def gs_cupy_jonas(V):
    for j in range(1, N):
        inner_prods = np.sum(V[j]*V, axis=1)
        V[j] = 2*V[j] - np.sum(inner_prods[:,None]*V, axis=0)


def gs_C_T(V):
    M, N = V.shape
    assert M < N, "M={0}, N={1}".format(M, N)
    for j in range(1, M):
        gs_c_lib.orthogonalise_vector(V, j)

def gs_C_omp_T(V):
    M, N = V.shape
    assert M < N, "M={0}, N={1}".format(M, N)
    for j in range(1, M):
        gs_c_lib.orthogonalise_vector_omp(V, j)

def gs_cuda_T(V):
    M, N = V.shape
    assert M < N, "M={0}, N={1}".format(M, N)
    gs_cuda_lib.orthogonalise(V)

def check_ort(A):
    print(np.dot(A[:,0], A[:,1]), np.dot(A[:,0], A[:,2]), np.dot(A[:,1], A[:,2]))

def check_ort_cp(A):
    print(cp.dot(A[:,0], A[:,1]), cp.dot(A[:,0], A[:,2]), cp.dot(A[:,1], A[:,2]))

def check_ort_cp_T(A):
    print(cp.dot(A[0,:], A[1,:]), cp.dot(A[0,:], A[2,:]), cp.dot(A[1,:], A[2,:]))


def bench_gs_numpy(mem_traffic=None):
    method_desc = "numpy"
    V = V_orig.copy()
    pre = time.time()
    gs_numpy(V)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V)
    print()
    return time_taken

def bench_gs_numpy_T(mem_traffic=None):
    method_desc = "numpy transposed"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_numpy_T(V_T)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

def bench_gs_numba(mem_traffic=None):
    method_desc = "numba"
    V = V_orig.copy()
    pre = time.time()
    gs_numba(V)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V)
    print()
    return time_taken

def bench_gs_numba_T(mem_traffic=None):
    method_desc = "numba transposed"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_numba_T(V_T)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

def bench_gs_cupy(mem_traffic=None):
    method_desc = "cupy"
    V_d = cp.array(V_orig.copy())
    pre = time.time()
    gs_cupy(V_d)
    check_ort_cp(V_d)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    print()
    return time_taken

def bench_gs_cupy_T(mem_traffic=None):
    method_desc = "cupy transposed"
    V_T_d = cp.array(V_orig.T.copy())
    pre = time.time()
    gs_cupy_T(V_T_d)
    check_ort_cp_T(V_T_d)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    print()
    return time_taken

def bench_gs_cupy_jonas(mem_traffic=None):
    method_desc = "cupy jonas"
    V_T_d = cp.array(V_orig.T.copy())
    pre = time.time()
    gs_cupy_jonas(V_T_d)
    check_ort_cp_T(V_T_d)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    print()
    return time_taken

def bench_gs_C_T(mem_traffic=None):
    method_desc = "C transposed"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_C_T(V_T)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

def bench_gs_C_omp_T(mem_traffic=None):
    method_desc = "C with OpenMP transposed"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_C_omp_T(V_T)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken

def bench_gs_cuda_T(mem_traffic=None):
    method_desc = "CUDA"
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_cuda_T(V_T)
    post = time.time()
    time_taken = post - pre
    msg = "Time taken {0}: {1:g} s".format(method_desc, time_taken)
    if mem_traffic is not None:
        effective_bw = mem_traffic / time_taken
        msg += " (effective bandwidth: {0:.0f} GB/s)".format(effective_bw/1E9)
    print(msg)
    check_ort(V_T.T)
    print()
    return time_taken


if __name__ == '__main__':
    M = int(1E6)
    N = 300
    sizeof_double = 8
    print("(M, N): {0}, array size: {1:g} MB".format(
        (M, N), M*N*sizeof_double/1E6))

    np.random.seed(1)
    V_orig = np.random.rand(M, N)


    mem_traffic_optimal = (N*N + N*N/2. + 2*N)*M*sizeof_double
    bytes_two_vectors = 2*M*sizeof_double
    print("Optimal memory traffic is {0:g} GB, assuming {1:.0f} MB cannot fit in cache".format(
        mem_traffic_optimal/1E9,
        bytes_two_vectors/1E6,
    ))
    print()
    if mem_traffic_optimal < 100E9:
        bench_gs_numpy()
        bench_gs_numpy_T()
        if has_numba:
            bench_gs_numba()
            bench_gs_numba_T()
        pass
    else:
        print("Skipping numpy benchmark since it will take forever")
    print()

    bench_gs_C_T(mem_traffic=mem_traffic_optimal)
    bench_gs_C_omp_T(mem_traffic=mem_traffic_optimal)
    print()

    if has_gs_cuda_lib:
        bench_gs_cuda_T(mem_traffic=mem_traffic_optimal)
        print()

    if has_cupy:
        bench_gs_cupy()
        bench_gs_cupy_T()
        bench_gs_cupy_jonas()
