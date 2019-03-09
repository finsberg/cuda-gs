On a single Tesla P100 (PCIe variant)

```
(base) [kgh@bigfacet 17:49:57 cuda-gs (master)*]$ numactl --cpubind=1 -- nvprof python gs_bench.py 
(M, N): (1000000, 300), array size: 2400 MB
Optimal memory traffic is 1084.8 GB, assuming 16 MB cannot fit in cache

Skipping numpy benchmark since it will take forever

==740== NVPROF is profiling process 740, command: python gs_bench.py
Time taken CUDA: 3.3375 s (effective bandwidth: 325 GB/s)
-1.0899725566559937e-11 3.679900828501559e-11 7.572609206363268e-12


==740== Profiling application: python gs_bench.py
==740== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   34.89%  959.03ms       299  3.2075ms  38.143us  6.3966ms  gs_dot_kernel(double*, int, int, int, int, double*, double*)
                   32.36%  889.61ms         1  889.61ms  889.61ms  889.61ms  [CUDA memcpy HtoD]
                   23.48%  645.50ms       299  2.1589ms  58.432us  4.2541ms  gs_subtract_projections(double*, int, int, double const *)
                    9.24%  253.95ms         1  253.95ms  253.95ms  253.95ms  [CUDA memcpy DtoH]
                    0.03%  826.30us       299  2.7630us  2.2080us  11.424us  gs_dot_reduce_kernel(int, double*, double*, double*)
      API calls:   51.85%  1.60262s       299  5.3599ms  85.578us  10.643ms  cudaStreamSynchronize
                   37.02%  1.14415s         2  572.08ms  254.08ms  890.08ms  cudaMemcpy
                   10.30%  318.38ms         1  318.38ms  318.38ms  318.38ms  cudaStreamCreate
                    0.39%  12.031ms         4  3.0078ms  2.3815ms  3.3171ms  cudaFree
                    0.24%  7.2647ms       897  8.0980us  6.3550us  61.545us  cudaLaunchKernel
                    0.11%  3.5149ms         4  878.73us  348.15us  2.4328ms  cudaMalloc
                    0.04%  1.2318ms        96  12.831us     210ns  508.80us  cuDeviceGetAttribute
                    0.04%  1.1980ms         1  1.1980ms  1.1980ms  1.1980ms  cudaGetDeviceProperties
                    0.01%  353.17us         1  353.17us  353.17us  353.17us  cuDeviceTotalMem
                    0.00%  147.33us       897     164ns     145ns     338ns  cudaPeekAtLastError
                    0.00%  112.09us         1  112.09us  112.09us  112.09us  cuDeviceGetName
                    0.00%  11.504us         1  11.504us  11.504us  11.504us  cudaStreamDestroy
                    0.00%  3.4820us         1  3.4820us  3.4820us  3.4820us  cuDeviceGetPCIBusId
                    0.00%  2.1550us         3     718ns     244ns  1.2430us  cuDeviceGetCount
                    0.00%  1.4050us         2     702ns     296ns  1.1090us  cuDeviceGet
                    0.00%     470ns         1     470ns     470ns     470ns  cuDeviceGetUuid
```
