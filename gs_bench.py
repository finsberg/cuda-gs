import time
import numba
import numpy as np
import cupy as cp

import gs_c_lib


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

def check_ort(A):
    print(np.dot(A[:,0], A[:,1]), np.dot(A[:,0], A[:,2]), np.dot(A[:,1], A[:,2]))

def check_ort_cp(A):
    print(cp.dot(A[:,0], A[:,1]), cp.dot(A[:,0], A[:,2]), cp.dot(A[:,1], A[:,2]))

def check_ort_cp_T(A):
    print(cp.dot(A[0,:], A[1,:]), cp.dot(A[0,:], A[2,:]), cp.dot(A[1,:], A[2,:]))


def bench_gs_numpy():
    V = V_orig.copy()
    pre = time.time()
    gs_numpy(V)
    post = time.time()
    print("Time taken numpy: %g s" % (post-pre))
    check_ort(V)
    print()

def bench_gs_numpy_T():
    V_T = V_orig.copy().T
    pre = time.time()
    gs_numpy_T(V_T)
    post = time.time()
    print("Time taken numpy transposed: %g s" % (post-pre))
    check_ort(V_T.T)
    print()

def bench_gs_numba():
    V = V_orig.copy()
    pre = time.time()
    gs_numba(V)
    post = time.time()
    print("Time taken numba: %g s" % (post-pre))
    check_ort(V)
    print()

def bench_gs_numba_T():
    V_T = V_orig.copy().T
    pre = time.time()
    gs_numba_T(V_T)
    post = time.time()
    print("Time taken numba transposed: %g s" % (post-pre))
    check_ort(V_T.T)
    print()

def bench_gs_cupy():
    V_d = cp.array(V_orig.copy())
    pre = time.time()
    gs_cupy(V_d)
    check_ort_cp(V_d)
    post = time.time()
    print("Time taken cupy: %g s" % (post-pre))
    print()

def bench_gs_cupy_T():
    V_T_d = cp.array(V_orig.copy().T)
    pre = time.time()
    gs_cupy_T(V_T_d)
    check_ort_cp_T(V_T_d)
    post = time.time()
    print("Time taken cupy T: %g s" % (post-pre))
    print()

def bench_gs_cupy_jonas():
    V_T_d = cp.array(V_orig.copy().T)
    pre = time.time()
    gs_cupy_jonas(V_T_d)
    check_ort_cp_T(V_T_d)
    post = time.time()
    print("Time taken cupy jonas: %g s" % (post-pre))
    print()

def bench_gs_C_T():
    V_T = V_orig.T.copy()
    pre = time.time()
    gs_C_T(V_T)
    post = time.time()
    print("Time taken C transposed: %g s" % (post-pre))
    check_ort(V_T.T)
    print()


if __name__ == '__main__':
    M = int(1E5)
    N = 100
    print("(M, N):", (M, N))

    np.random.seed(1)
    V_orig = np.random.rand(M, N)

    mem_traffic_optimal = (N*N + N*N/2. + N)*M*8
    print("Optimal memory traffic {0:g} GB".format(mem_traffic_optimal/1E9))
    print()
    if True or mem_traffic_optimal < 10E9:
        bench_gs_numpy()
        bench_gs_numpy_T()
        bench_gs_numba()
        bench_gs_numba_T()

    else:
        print("Skipping numpy benchmark since it will take forever")
    print()

    bench_gs_C_T()
    print()

    bench_gs_cupy()
    bench_gs_cupy_T()
    bench_gs_cupy_jonas()
