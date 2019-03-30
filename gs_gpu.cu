#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>
#include <cuda.h>

#ifdef HAS_NCCL
#include <nccl.h>
#endif

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

#ifdef HAS_NCCL
#define checkNccl(retval) __checkNccl(retval, __FILE__, __LINE__)

inline void __checkNccl(ncclResult_t retval, const char *file, const int line)
{
    if (retval != ncclSuccess) {
        printf("checkNccl error at %s:%i: return value: %d\n",
               file, line, retval);
        exit(EXIT_FAILURE);
    }
    return;
}
#endif

enum init_returncode {
    INIT_SUCCESS,
    INIT_NO_DEVICES,
    INIT_INSUFFICIENT_MEMORY,
    INIT_MISSING_OPENMP,
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
}

__global__ void gs_dot_reduce_kernel_combine_frac(int dot_prod_parts_row_length,
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
    printf("dotreduce prereduce (%d, %d): uu: %8g  uv: %8g\n",
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

__global__ void gs_dot_reduce_kernel_uu_and_uv(int dot_prod_parts_row_length,
    double *dot_prod_uu_parts, double *dot_prod_uv_parts,
    double *dot_prod_uu_local, double *dot_prod_uv_local)
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
    printf("dotreduce prereduce (%d, %d): uu: %8g  uv: %8g\n",
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
        dot_prod_uu_local[u_ind] = dot_prod_uu_part;
        dot_prod_uv_local[u_ind] = dot_prod_uv_part;
    }
}

__global__ void gs_subtract_projections(double *V, int N, int v_ind,
    const double *__restrict dot_prod_uu, const double *__restrict dot_prod_uv)
{
    int col_offset = blockIdx.x*blockDim.x + threadIdx.x;
    if (col_offset < N) {
        double proj = 0;
        double *u = V;
        for (int i = 0; i < v_ind; i++) {
            if (dot_prod_uu[i] > TOL_SQUARED) {
                proj += dot_prod_uv[i] / dot_prod_uu[i] * u[col_offset];
            }
            u += N;
        }
        V[v_ind*N + col_offset] -= proj;
    }
}

__global__ void gs_subtract_projections_dot_frac(double *V, int N, int v_ind,
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
#ifdef HAS_NCCL
    int N_global;
    int N_start;
    int N_end;
    ncclComm_t comm;
    ncclUniqueId comm_id;
    double *dot_prod_uu_local;  // length M
    double *dot_prod_uv_local;  // length M
    double *dot_prod_uu_global;  // length M
    double *dot_prod_uv_global;  // length M
    size_t dot_prod_size;
#endif
};

struct gs_node_ctx {
    int device_count;
    gs_data *data;
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

#ifdef HAS_NCCL
int gs_init_nccl(struct gs_node_ctx** node_ctx_ptr, int M, int N_total,
                 int* device_count_ptr)
{
    int device_count = 0;
    {
        cudaError_t error;
        error = cudaGetDeviceCount(&device_count);
        if (error == cudaErrorNoDevice) {
            return INIT_NO_DEVICES;
        }
        checkCuda(error);
    }
    *device_count_ptr = device_count;
    //printf("Using %d devices.\n", device_count);

    int N_limits[device_count+1];
    for (int i= 0; i <= device_count; i++) {
        N_limits[i] = N_total * i / device_count;
    }

    int insufficient_memory = 0;
    int threads_seen = 0;
    #pragma omp parallel num_threads(device_count)
    {
        int thread_id = omp_get_thread_num();
        int device_num = thread_id;
        cudaDeviceProp deviceProp;
        checkCuda(cudaGetDeviceProperties(&deviceProp, device_num));
        int num_SMs = deviceProp.multiProcessorCount;
        size_t global_memory_size = deviceProp.totalGlobalMem;

        //int M_local = M;
        int N_local = N_limits[thread_id+1] - N_limits[thread_id];

        // compute global memory usage
        size_t V_size = M*N_local*sizeof(double);
        size_t dot_prod_size = M*sizeof(double);

        int max_threadblocks_per_row_dot_kernel = threadblocks_per_row_dot_kernel(num_SMs, 1);
        size_t dot_prod_buffer_size = M * max_threadblocks_per_row_dot_kernel * sizeof(double);

        size_t required_memory_size = V_size + 2*dot_prod_size + 2*dot_prod_buffer_size;

        #pragma omp critical
        {
            if (global_memory_size < required_memory_size) {
                printf("error: device %d would use %.0f / %.0f MB of global memory\n",
                    device_num, required_memory_size/1E6, global_memory_size/1E6
                );
                insufficient_memory = 1;
            }
            threads_seen++;
        }
    }
    if (insufficient_memory) {
        return INIT_INSUFFICIENT_MEMORY;
    }
    if (threads_seen == 1 && device_count != 1) {
        return INIT_MISSING_OPENMP;
    }

    struct gs_node_ctx *node_ctx = (struct gs_node_ctx *) malloc(sizeof(struct gs_node_ctx));
    //struct gs_data *global_data = (struct gs_data *) malloc(sizeof(struct gs_data)*device_count);
    node_ctx->device_count = device_count;
    node_ctx->data = (struct gs_data *) malloc(sizeof(struct gs_data)*device_count);
    ncclUniqueId comm_id;
    ncclGetUniqueId(&comm_id);

    #pragma omp parallel num_threads(device_count)
    {
        int thread_id = omp_get_thread_num();
        int device_num = thread_id;
        cudaSetDevice(device_num);

        cudaDeviceProp deviceProp;
        checkCuda(cudaGetDeviceProperties(&deviceProp, device_num));
        int num_SMs = deviceProp.multiProcessorCount;


        int N_local = N_limits[device_num+1] - N_limits[device_num];

        size_t V_size = M*N_local*sizeof(double);
        size_t dot_prod_size = M*sizeof(double);

        int max_threadblocks_per_row_dot_kernel = threadblocks_per_row_dot_kernel(num_SMs, 1);
        size_t dot_prod_buffer_size = M * max_threadblocks_per_row_dot_kernel * sizeof(double);

        struct gs_data *mydata = node_ctx->data + device_num;
#ifdef DEBUG
        printf("About to call ncclCommInitRank(%p, %d, %d, %d)\n",
            &mydata->comm, device_count, comm_id, device_num);
#endif
        ncclCommInitRank(&mydata->comm, device_count, comm_id, device_num);
        mydata->comm_id = comm_id;
        mydata->N_start = N_limits[device_num];
        mydata->N_end = N_limits[device_num+1];
        mydata->M = M;
        mydata->N = N_local;
        mydata->N_global = N_total;
        mydata->num_SMs = num_SMs;
        mydata->dot_prod_size = dot_prod_size;

        checkCuda(cudaStreamCreate(&mydata->stream));

        checkCuda(cudaMalloc(&mydata->V_d, V_size));
        checkCuda(cudaMalloc(&mydata->dot_prod_uu_local, dot_prod_size));
        checkCuda(cudaMalloc(&mydata->dot_prod_uv_local, dot_prod_size));
        checkCuda(cudaMalloc(&mydata->dot_prod_buffer_uu, dot_prod_buffer_size));
        checkCuda(cudaMalloc(&mydata->dot_prod_buffer_uv, dot_prod_buffer_size));
        mydata->dot_prod_uu_global = mydata->dot_prod_buffer_uu;
        mydata->dot_prod_uv_global = mydata->dot_prod_buffer_uv;
    }
    //*data_ptr = global_data;
    *node_ctx_ptr = node_ctx;
#ifdef DEBUG
    printf("Init done\n");
#endif

    return INIT_SUCCESS;
}

void gs_cleanup_nccl(struct gs_node_ctx* node_ctx)
{
    int device_count = node_ctx->device_count;
    #pragma omp parallel num_threads(device_count)
    {
        int thread_id = omp_get_thread_num();
        int device_num = thread_id;
        cudaSetDevice(device_num);

        struct gs_data *data = node_ctx->data + device_num;
        checkNccl(ncclCommDestroy(data->comm));
        checkCuda(cudaFree(data->V_d));
        checkCuda(cudaFree(data->dot_prod_buffer_uu));
        checkCuda(cudaFree(data->dot_prod_buffer_uv));
        checkCuda(cudaFree(data->dot_prod_uu_local));
        checkCuda(cudaFree(data->dot_prod_uv_local));
        checkCuda(cudaStreamDestroy(data->stream));
    }
    free(node_ctx->data);
    free(node_ctx);
}

void gs_copy_from_host_nccl(struct gs_node_ctx* node_ctx, double *V_h)
{
    int device_count = node_ctx->device_count;
    #pragma omp parallel num_threads(device_count)
    {
        int thread_id = omp_get_thread_num();
        int device_num = thread_id;
        cudaSetDevice(device_num);

        struct gs_data *data = node_ctx->data + device_num;
        int M = data->M;
        int N = data->N;
        int N_local = data->N_end - data->N_start;
        size_t my_rowsize = N_local*sizeof(double);

        size_t hostmem_offset = data->N_start;
        size_t devicemem_offset = 0;
        for (int row = 0; row < M; row++) {
#ifdef DEBUG
            printf(" on %d: memcpy from host. row=%d. Copying %lu bytes from %p to %p.\n",
            device_num, row, my_rowsize, V_h+hostmem_offset, data->V_d+devicemem_offset);
#endif
            checkCuda(cudaMemcpyAsync(
                data->V_d+devicemem_offset, V_h+hostmem_offset, my_rowsize,
                cudaMemcpyHostToDevice, data->stream
            ));
            hostmem_offset += N;
            devicemem_offset += N_local;
        }
        checkCuda(cudaStreamSynchronize(data->stream));
    }
}

void gs_copy_to_host_nccl(struct gs_node_ctx* node_ctx, double *V_h)
{
    int device_count = node_ctx->device_count;
    #pragma omp parallel num_threads(device_count)
    {
        int thread_id = omp_get_thread_num();
        int device_num = thread_id;
        cudaSetDevice(device_num);

        struct gs_data *data = node_ctx->data + device_num;
        int M = data->M;
        int N = data->N_global;
        int N_local = data->N_end - data->N_start;
        size_t my_rowsize = N_local*sizeof(double);

        size_t hostmem_offset = data->N_start;
        size_t devicemem_offset = 0;
        for (int row = 0; row < M; row++) {
#ifdef DEBUG
            printf(" on %d: memcpy to host. row=%d. Copying %lu bytes from %p to %p.\n",
            device_num, row, my_rowsize, data->V_d+devicemem_offset, V_h+hostmem_offset);
#endif
            checkCuda(cudaMemcpyAsync(
                V_h+hostmem_offset, data->V_d+devicemem_offset, my_rowsize,
                cudaMemcpyDeviceToHost, data->stream
            ));
            hostmem_offset += N;
            devicemem_offset += N_local;
        }
        checkCuda(cudaStreamSynchronize(data->stream));
    }
}

void gs_orthogonalise_vector_nccl(struct gs_node_ctx* node_ctx, int new_vec_ind)
{
    int device_count = node_ctx->device_count;
    #pragma omp parallel num_threads(device_count)
    {
        int thread_id = omp_get_thread_num();
        int device_num = thread_id;
        cudaSetDevice(device_num);

        struct gs_data *data = node_ctx->data + device_num;

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

            dim3 grid_dim(threadblocks_x, threadblocks_y);
            dim3 threadblock_dim(threadblock_size);

            gs_dot_kernel<<<grid_dim, threadblock_dim, 0, data->stream>>>(
                V_d, N, new_vec_ind, elements_per_thread, dot_prod_parts_row_length,
                data->dot_prod_buffer_uu, data->dot_prod_buffer_uv
            );
            checkCuda(cudaPeekAtLastError());

            // reduce to single value per row
            gs_dot_reduce_kernel_uu_and_uv<<<new_vec_ind, 32, 0, data->stream>>>(
                dot_prod_parts_row_length,
                data->dot_prod_buffer_uu, data->dot_prod_buffer_uv,
                data->dot_prod_uu_local, data->dot_prod_uv_local
            );
            checkCuda(cudaPeekAtLastError());
        }

        // reduce dot products
        ncclAllReduce(data->dot_prod_uu_local, data->dot_prod_uu_global,
                      data->dot_prod_size, ncclDouble, ncclSum, data->comm, data->stream);
        ncclAllReduce(data->dot_prod_uv_local, data->dot_prod_uv_global,
                      data->dot_prod_size, ncclDouble, ncclSum, data->comm, data->stream);

        // subtract projections
        {
            int threadblock_size = 32;
            int threadblocks = CEIL_DIV(N, threadblock_size);

            dim3 grid_dim(threadblocks);
            dim3 threadblock_dim(threadblock_size);
            gs_subtract_projections<<<grid_dim, threadblock_dim, 0, data->stream>>>(
                V_d, N, new_vec_ind,
                data->dot_prod_uu_global, data->dot_prod_uv_global
            );
            checkCuda(cudaPeekAtLastError());
        }

        checkCuda(cudaStreamSynchronize(data->stream));
    }
}
#endif // HAS_NCCL

int gs_init(struct gs_data** data_ptr, int M, int N, int* device_count_ptr)
{
    int device_count;
    {
        cudaError_t error;
        error = cudaGetDeviceCount(&device_count);
        if (error == cudaErrorNoDevice) {
            return INIT_NO_DEVICES;
        }
        checkCuda(error);
    }
    *device_count_ptr = device_count;

    int device_num = 0;
    cudaDeviceProp deviceProp;
    checkCuda(cudaGetDeviceProperties(&deviceProp, device_num));
    int num_SMs = deviceProp.multiProcessorCount;
    size_t global_memory_size = deviceProp.totalGlobalMem;

    // compute global memory usage
    size_t V_size = M*N*sizeof(double);
    size_t dot_prod_frac_size = M*sizeof(double);

    int max_threadblocks_per_row_dot_kernel = threadblocks_per_row_dot_kernel(num_SMs, 1);
    size_t dot_prod_buffer_size = M * max_threadblocks_per_row_dot_kernel * sizeof(double);

    size_t required_memory_size = V_size + dot_prod_frac_size + 2*dot_prod_buffer_size;

    if (global_memory_size < required_memory_size) {
        return INIT_INSUFFICIENT_MEMORY;
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
        checkCuda(cudaMemcpy(dot_prod_buffer_uu, data->dot_prod_buffer_uu, dot_prod_buffer_activesize, cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(dot_prod_buffer_uv, data->dot_prod_buffer_uv, dot_prod_buffer_activesize, cudaMemcpyDeviceToHost));
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
        gs_dot_reduce_kernel_combine_frac<<<new_vec_ind, 32, 0, data->stream>>>(
            dot_prod_parts_row_length,
            data->dot_prod_buffer_uu, data->dot_prod_buffer_uv,
            data->dot_prod_frac
        );
        checkCuda(cudaPeekAtLastError());
    }

#ifdef DEBUG
    double dot_prod_frac[new_vec_ind];
    checkCuda(cudaMemcpy(dot_prod_frac, data->dot_prod_frac, sizeof(double)*new_vec_ind, cudaMemcpyDeviceToHost));
    for (int i = 0; i < new_vec_ind; i++) {
        printf("%3d: %g\n", i, dot_prod_frac[i]);
    }
#endif

    // subtract projections
    {
        int threadblock_size = 32;
        int threadblocks = CEIL_DIV(N, threadblock_size);

        dim3 grid_dim(threadblocks);
        dim3 threadblock_dim(threadblock_size);
        gs_subtract_projections_dot_frac<<<grid_dim, threadblock_dim, 0, data->stream>>>(
            V_d, N, new_vec_ind, data->dot_prod_frac
        );
        checkCuda(cudaPeekAtLastError());
    }

    checkCuda(cudaStreamSynchronize(data->stream));
}

} // end extern "C"
