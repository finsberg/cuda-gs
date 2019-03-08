#include <stdio.h>
#include <assert.h>

#define TOL 1E-14
#define TOL_SQUARED (TOL*TOL)

/* orthogonalise V[new_vec_ind,:] with respect to V[:new_vec_ind,:],
  V is of size M x N, and new_vec_ind < M */
void gs_orthogonalise_vector(double *V, int M, int N, int new_vec_ind)
{
    assert(new_vec_ind < M);

    // let v be V[new_vec_ind,:] and u be V[i,:]
    double *v = V + N*new_vec_ind;
    double *u = V;

    for (int i = 0; i < new_vec_ind; i++) {
        // compute dot(u, u) and (u, v)
        double inner_prod_uu = 0;
        double inner_prod_uv = 0;
        for (int k = 0; k < N; k++) {
            inner_prod_uu += u[k]*u[k];
            inner_prod_uv += u[k]*v[k];
        }

        // subtract projection unless u is a null vector
        if (inner_prod_uu > TOL_SQUARED) {
            for (int k = 0; k < N; k++) {
                v[k] -= inner_prod_uv / inner_prod_uu * u[k];
            }
        }

        // move u pointer to next vector
        u += N;
    }
}
