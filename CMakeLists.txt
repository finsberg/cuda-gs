cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project (cuda-gs LANGUAGES C CUDA)

add_library(gs SHARED gs.c)
add_library(gs_gpu SHARED gs_gpu.cu)

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OpenMP found")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_C_FLAGS}")
    target_link_libraries(gs OpenMP::OpenMP_C)
    target_link_libraries(gs_gpu ${OpenMP})
endif()

find_library(NCCL nccl)
if (NCCL)
    message("NCCL found")
    add_definitions(-DHAS_NCCL)
    target_link_libraries(gs_gpu ${NCCL})
endif ()
