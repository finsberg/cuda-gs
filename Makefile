LIBNAME_CPU = libgs
LIBNAME_GPU = libgs_gpu
LIBFILE_CPU = $(LIBNAME_CPU).so
LIBFILE_GPU = $(LIBNAME_GPU).so
LIB_CPU_SOURCES = gs.c
LIB_CPU_OBJECTS = $(LIB_CPU_SOURCES:.c=.o)
LIB_GPU_SOURCES = gs_gpu.cu
LIB_GPU_OBJECTS = $(LIB_GPU_SOURCES:.cu=.o)

NVCC = nvcc
NVCCFLAGS = -O3 -Xcompiler -fpic
NVCCFLAGS += -DHAS_NCCL -Xcompiler -fopenmp
LDFLAGS_GPU = -L/usr/local/cuda/lib -L/usr/local/cuda/lib64
LDLIBS_GPU = -lcudart
LDLIBS_GPU += -lnccl

CFLAGS = -Wall -O3 -fopenmp -fpic -march=native

all: $(LIBFILE_CPU) $(LIBFILE_GPU)

$(LIBFILE_CPU): $(LIB_CPU_OBJECTS)
	$(CC) $(CFLAGS) -shared -o $@ $^

$(LIBFILE_GPU): $(LIB_GPU_OBJECTS)
	$(NVCC) -shared -o $@ $^ $(LDFLAGS_GPU) $(LDLIBS_GPU)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	$(RM) $(LIB_CPU_OBJECTS) $(LIB_GPU_OBJECTS) $(LIBFILE_CPU) $(LIBFILE_GPU)
