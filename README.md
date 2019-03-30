This library implements Gram-Schmidt in numpy, cupy, numba, C (serially and with OpenMP) and CUDA (with a single GPU or multiple GPUs using NCCL).

It demonstrates that there is a significant performance difference between a typical Python implementation in numpy (or numba) for CPU and cupy for GPU, compared to a proper implementation in C (for CPU) or CUDA (for GPU).

## Requirements
* Python 3 with numpy

### Optional
* numba
* cupy
* A C compiler with OpenMP support is required for the C library
* `nvcc` is required to build the CUDA library
* [NCCL](https://github.com/NVIDIA/nccl) is required for multiple GPU support.

## Building
The required libraries can be built with either a plain GNU makefile or CMake (which should be better at finding libraries).

### Building with a plain GNU makefile
```
make -f Makefile.GNU
```

### Building with CMake
```
cmake .
make
```

### Running
The benchmark can be run with
```
python gs_bench.py
```
