#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>

#define TOL 1E-14
#define TOL_SQUARED (TOL*TOL)

//#define DEBUG

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
    if (error != cudaSuccess) {
        printf("checkCuda error at %s:%i: %s\n", file, line,
               cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }
    return;
}

enum init_returncode {
    INIT_SUCCESS,
    INIT_INSUFFIENT_MEMORY
};

__global__ void gs_dot_kernel(double *V, int N, int v_ind, int elements_per_thread,
    int dot_prod_parts_row_length, double *dot_prod_uu_parts, double *dot_prod_uv_parts)
{
    int u_ind = blockIdx.y;
    int i = blockIdx.x*blockDim.x*elements_per_thread + threadIdx.x;

    double *u = V + N*u_ind;
    double *v = V + N*v_ind;

    double dot_prod_uu_part = 0;
    double dot_prod_uv_part = 0;
    const int step = blockDim.x;
    for (int k = 0; k < elements_per_thread; k++) {
        if (i < N) {
            dot_prod_uu_part += u[i]*u[i];
            dot_prod_uv_part += u[i]*v[i];
        }
        i += step;
    }
#if 0
    int dot_prod_parts_ind = u_ind*dot_prod_parts_row_length + blockIdx.x*blockDim.x + threadIdx.x;
    dot_prod_uu_parts[dot_prod_parts_ind] = dot_prod_uu_part;
    dot_prod_uv_parts[dot_prod_parts_ind] = dot_prod_uv_part;
#else
    // reduce within warp
    dot_prod_uu_part += __shfl_down_sync(0xffffffff, dot_prod_uu_part, 16, 32);
    dot_prod_uu_part += __shfl_down_sync(0xffff0000, dot_prod_uu_part,  8, 32);
    dot_prod_uu_part += __shfl_down_sync(0xff000000, dot_prod_uu_part,  4, 32);
    dot_prod_uu_part += __shfl_down_sync(0xf0000000, dot_prod_uu_part,  2, 32);
    dot_prod_uu_part += __shfl_down_sync(0xc0000000, dot_prod_uu_part,  1, 32);

    dot_prod_uv_part += __shfl_down_sync(0xffffffff, dot_prod_uv_part, 16, 32);
    dot_prod_uv_part += __shfl_down_sync(0xffff0000, dot_prod_uv_part,  8, 32);
    dot_prod_uv_part += __shfl_down_sync(0xff000000, dot_prod_uv_part,  4, 32);
    dot_prod_uv_part += __shfl_down_sync(0xf0000000, dot_prod_uv_part,  2, 32);
    dot_prod_uv_part += __shfl_down_sync(0xc0000000, dot_prod_uv_part,  1, 32);

    if (threadIdx.x == 0) {
        int dot_prod_parts_ind = u_ind*dot_prod_parts_row_length + blockIdx.x;
        dot_prod_uu_parts[dot_prod_parts_ind] = dot_prod_uu_part;
        dot_prod_uv_parts[dot_prod_parts_ind] = dot_prod_uv_part;
    }
#endif
}

__global__ void gs_dot_reduce_kernel(int dot_prod_parts_row_length,
    double *dot_prod_uu_parts, double *dot_prod_uv_parts,
    double *dot_prod_frac)
{
    int u_ind = blockIdx.x;
    //int i = blockIdx.x*blockDim.x*elements_per_thread + threadIdx.x;
    int steps = CEIL_DIV(dot_prod_parts_row_length, blockDim.x);
    int i = u_ind * dot_prod_parts_row_length + threadIdx.x;
    double dot_prod_uu_part = 0;
    double dot_prod_uv_part = 0;
    int max_i = (u_ind + 1) * dot_prod_parts_row_length;
    for (int k = 0; k < steps; k++) {
        if (i < max_i) {
            dot_prod_uu_part += dot_prod_uu_parts[i];
            dot_prod_uv_part += dot_prod_uv_parts[i];
        }
        i += blockDim.x;
    }

#ifdef DEBUG
    printf("dotreduce prerelease (%d, %d): uu: %8g  uv: %8g\n",
        blockIdx.x, threadIdx.x, dot_prod_uu_part, dot_prod_uv_part);
#endif

    // reduce within warp
    dot_prod_uu_part += __shfl_down_sync(0xffffffff, dot_prod_uu_part, 16, 32);
    dot_prod_uu_part += __shfl_down_sync(0xffff0000, dot_prod_uu_part,  8, 32);
    dot_prod_uu_part += __shfl_down_sync(0xff000000, dot_prod_uu_part,  4, 32);
    dot_prod_uu_part += __shfl_down_sync(0xf0000000, dot_prod_uu_part,  2, 32);
    dot_prod_uu_part += __shfl_down_sync(0xc0000000, dot_prod_uu_part,  1, 32);

    dot_prod_uv_part += __shfl_down_sync(0xffffffff, dot_prod_uv_part, 16, 32);
    dot_prod_uv_part += __shfl_down_sync(0xffff0000, dot_prod_uv_part,  8, 32);
    dot_prod_uv_part += __shfl_down_sync(0xff000000, dot_prod_uv_part,  4, 32);
    dot_prod_uv_part += __shfl_down_sync(0xf0000000, dot_prod_uv_part,  2, 32);
    dot_prod_uv_part += __shfl_down_sync(0xc0000000, dot_prod_uv_part,  1, 32);

    // now all 32 parts are summed up
    // let thread 0 write back the final value
    if (threadIdx.x == 0) {
#ifdef DEBUG
        printf("dotreduce (%d, %d): uu: %8g  uv: %8g\n",
            blockIdx.x, threadIdx.x, dot_prod_uu_part, dot_prod_uv_part);
#endif
        if (fabs(dot_prod_uu_part) > TOL_SQUARED) {
            dot_prod_frac[u_ind] = dot_prod_uv_part / dot_prod_uu_part;
        } else {
#ifdef DEBUG
            printf("dotreduce (%d, %d): %g\n", blockIdx.x, threadIdx.x, dot_prod_uu_part);
#endif
            dot_prod_frac[u_ind] = 0;
        }
    }
}

__global__ void gs_subtract_projections(double *V, int N, int v_ind,
    const double *__restrict dot_prod_frac)
{
    int col_offset = blockIdx.x*blockDim.x + threadIdx.x;
    if (col_offset < N) {
        double proj = 0;
        double *u = V;
        for (int i = 0; i < v_ind; i++) {
            if (dot_prod_frac[i] != 0) {
                proj += dot_prod_frac[i]*u[col_offset];
            }
            u += N;
        }
        V[v_ind*N + col_offset] -= proj;
    }
}



extern "C" {

struct gs_data {
    double *V_d;                    // length M*N
    double *dot_prod_buffer_uu;
    double *dot_prod_buffer_uv;
    double *dot_prod_frac;         // length M
    int M;
    int N;
    int num_SMs;                    // # of stream multiprocessors
    cudaStream_t stream;
};

extern int threadblocks_per_row_dot_kernel(int num_SMs, int threadblocks_y)
{
    int threadblocks_target = 32 * num_SMs;  // should be set experimentally
#ifdef DEBUG
    threadblocks_target = 10;
#endif
    int threadblocks_x = CEIL_DIV(threadblocks_target, threadblocks_y);
    return threadblocks_x;
}

int gs_init(struct gs_data** data_ptr, int M, int N)
{
    int device_num = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_num);
    int num_SMs = deviceProp.multiProcessorCount;
    size_t global_memory_size = deviceProp.totalGlobalMem;

    // compute global memory usage
    size_t V_size = M*N*sizeof(double);
    size_t dot_prod_frac_size = M*sizeof(double);

    int max_threadblocks_per_row_dot_kernel = threadblocks_per_row_dot_kernel(num_SMs, 1);
    size_t dot_prod_buffer_size = M * max_threadblocks_per_row_dot_kernel * sizeof(double);

    size_t required_memory_size = V_size + dot_prod_frac_size + 2*dot_prod_buffer_size;

    if (global_memory_size < required_memory_size) {
        return INIT_INSUFFIENT_MEMORY;
    }

    struct gs_data *data = (struct gs_data *) malloc(sizeof(struct gs_data));
    data->M = M;
    data->N = N;
    data->num_SMs = num_SMs;

    checkCuda(cudaStreamCreate(&data->stream));

    checkCuda(cudaMalloc(&data->V_d, V_size));
    checkCuda(cudaMalloc(&data->dot_prod_frac, dot_prod_frac_size));
    checkCuda(cudaMalloc(&data->dot_prod_buffer_uu, dot_prod_buffer_size));
    checkCuda(cudaMalloc(&data->dot_prod_buffer_uv, dot_prod_buffer_size));

    *data_ptr = data;

    return INIT_SUCCESS;
}

void gs_cleanup(struct gs_data* data)
{
    checkCuda(cudaFree(data->V_d));
    checkCuda(cudaFree(data->dot_prod_buffer_uu));
    checkCuda(cudaFree(data->dot_prod_buffer_uv));
    checkCuda(cudaFree(data->dot_prod_frac));
    checkCuda(cudaStreamDestroy(data->stream));
    free(data);
}

void gs_copy_from_host(struct gs_data* data, double *V_h)
{
    size_t V_size = data->M * data->N * sizeof(double);
    checkCuda(cudaMemcpy(data->V_d, V_h, V_size, cudaMemcpyHostToDevice));
}

void gs_copy_to_host(struct gs_data* data, double *V_h)
{
    size_t V_size = data->M*data->N*sizeof(double);
    checkCuda(cudaMemcpy(V_h, data->V_d, V_size, cudaMemcpyDeviceToHost));
}

void gs_orthogonalise_vector(struct gs_data* data, int new_vec_ind)
{
    const int M = data->M;
    assert(new_vec_ind < M);
    const int N = data->N;
    double *V_d = data->V_d;
    // compute dot products
    {
        int threadblocks_y = new_vec_ind;
        int threadblocks_x = threadblocks_per_row_dot_kernel(data->num_SMs, threadblocks_y);

        int threadblock_size = 32;
        //int threadblock_size = 64;
        //int threadblock_size = 128;
        int elements_per_thread = CEIL_DIV(N, threadblocks_x*threadblock_size);

        int dot_prod_parts_row_length = threadblocks_x;
#ifdef DEBUG
        printf("dot product         on line %3d ((%3d, %3d), %3d)  %7d\n",
            new_vec_ind, threadblocks_x, threadblocks_y, threadblock_size, elements_per_thread);
#endif
        dim3 grid_dim(threadblocks_x, threadblocks_y);
        dim3 threadblock_dim(threadblock_size);

        gs_dot_kernel<<<grid_dim, threadblock_dim, 0, data->stream>>>(
            V_d, N, new_vec_ind, elements_per_thread, dot_prod_parts_row_length,
            data->dot_prod_buffer_uu, data->dot_prod_buffer_uv
        );
        checkCuda(cudaPeekAtLastError());

#ifdef DEBUG
    {
        double dot_prod_buffer_uu[dot_prod_parts_row_length*new_vec_ind];
        double dot_prod_buffer_uv[dot_prod_parts_row_length*new_vec_ind];
        size_t dot_prod_buffer_activesize = dot_prod_parts_row_length*new_vec_ind*sizeof(double);
        cudaMemcpy(dot_prod_buffer_uu, data->dot_prod_buffer_uu, dot_prod_buffer_activesize, cudaMemcpyDeviceToHost);
        cudaMemcpy(dot_prod_buffer_uv, data->dot_prod_buffer_uv, dot_prod_buffer_activesize, cudaMemcpyDeviceToHost);
        for (int i = 0; i < new_vec_ind; i++) {
            printf("[%3d]: \n", i);
            for (int j = 0; j < dot_prod_parts_row_length; j++) {
                printf("  [%3d]: uu: %8g  uv: %8g\n",
                    j,
                    dot_prod_buffer_uu[i*dot_prod_parts_row_length+j],
                    dot_prod_buffer_uv[i*dot_prod_parts_row_length+j]
                );
            }
        }
    }

#endif

        // reduce to single value per row
        gs_dot_reduce_kernel<<<new_vec_ind, 32, 0, data->stream>>>(
            dot_prod_parts_row_length,
            data->dot_prod_buffer_uu, data->dot_prod_buffer_uv,
            data->dot_prod_frac
        );
        checkCuda(cudaPeekAtLastError());
    }

#ifdef DEBUG
    double dot_prod_frac[new_vec_ind];
    cudaMemcpy(dot_prod_frac, data->dot_prod_frac, sizeof(double)*new_vec_ind, cudaMemcpyDeviceToHost);
    for (int i = 0; i < new_vec_ind; i++) {
        printf("%3d: %g\n", i, dot_prod_frac[i]);
    }
#endif

    // subtract projections
    {
        int threadblock_size = 32;
        int threadblocks = CEIL_DIV(N, threadblock_size);
        /*
        printf("subtract projection on line %3d (%d, %d)\n",
            new_vec_ind, threadblocks, threadblock_size);
            */

        dim3 grid_dim(threadblocks);
        dim3 threadblock_dim(threadblock_size);
        gs_subtract_projections<<<grid_dim, threadblock_dim, 0, data->stream>>>(
            V_d, N, new_vec_ind, data->dot_prod_frac
        );
        checkCuda(cudaPeekAtLastError());
    }

    cudaStreamSynchronize(data->stream);
}

} // end extern "C"
