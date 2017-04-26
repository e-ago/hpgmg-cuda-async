# find MPI compiler
CC=`which mpicc`
#CC=`which mpiicc`
CXX=`which mpic++`

# find NVCC compiler
NVCC=`which nvcc`

# set gpu architectures to compile for
#CUDA_ARCH+="-gencode code=sm_35,arch=compute_35 "
#CUDA_ARCH+="-gencode code=sm_50,arch=compute_50 "
CUDA_ARCH+="-gencode code=sm_60,arch=compute_60 "

# main tile size
OPTS+="-DBLOCKCOPY_TILE_I=32 "
OPTS+="-DBLOCKCOPY_TILE_J=4 "
OPTS+="-DBLOCKCOPY_TILE_K=8 "

# special tile size for boundary conditions
OPTS+="-DBOUNDARY_TILE_I=64 "
OPTS+="-DBOUNDARY_TILE_J=16 "
OPTS+="-DBOUNDARY_TILE_K=16 "

# host level threshold: number of grid elements
#OPTS+="-DHOST_LEVEL_SIZE_THRESHOLD=10000 "
#become environment variable

# max number of solves after warmup
OPTS+="-DMAX_SOLVES=100 "


# host level threshold: number of grid elements
OPTS+="-DHOST_LEVEL_SIZE_THRESHOLD=10000 "

# max number of solves after warmup
#OPTS+="-DMAX_SOLVES=10 "

# unified memory allocation options
OPTS+="-DCUDA_UM_ALLOC "
#cudaHostAlloc
OPTS+="-DCUDA_UM_ZERO_COPY "

# MPI buffers allocation policy
#cudaHostAlloc
OPTS+="-DMPI_ALLOC_ZERO_COPY "
#cudaMalloc
#OPTS+="-DMPI_ALLOC_PINNED "


:<<AUTHOR_COMMENTS
OPTS+="-DMPI_ALLOC_ZERO_COPY " - uses cudaMallocHost to allocate MPI buffers
OPTS+="-DMPI_ALLOC_PINNED "  - uses cudaMalloc or malloc to allocate MPI buffers (depends on the level size)
If you comment out both options it will use cudaMallocManaged to allocate MPI buffers
AUTHOR_COMMENTS

#With gpu direct async, unified memory cannot be used
#comment if you want to use MPI+Unified Memory
#OPTS+="-DCUDA_DISABLE_UNIFIED_MEMORY "

# stencil optimizations
OPTS+="-DUSE_REG "
OPTS+="-DUSE_TEX "
#OPTS+="-DUSE_SHM "

# GSRB smoother options
#OPTS+="-DGSRB_FP "
#OPTS+="-DGSRB_STRIDE2 "
#OPTS+="-DGSRB_BRANCH "
#OPTS+="-DGSRB_OOP "

# tools
#OPTS+="-DUSE_PROFILE "
#OPTS+="-DUSE_NVTX "
#OPTS+="-DUSE_ERROR "

# override MVAPICH flags for C++
OPTS+="-DMPICH_IGNORE_CXX_SEEK "
OPTS+="-DMPICH_SKIP_MPICXX "  

#libmp flags
OPTS+="-DENABLE_EXCHANGE_BOUNDARY_COMM=1 "
OPTS+="-DENABLE_RESTRICTION_COMM=1 "
OPTS+="-DENABLE_RESTRICTION_ASYNC=1 "
OPTS+="-DENABLE_INTERPOLATION_COMM=1 "
OPTS+="-DENABLE_INTERPOLATION_ASYNC=1 "

:<<FLAGS_TESTS
#BUG HERE: deadlock
OPTS+="-DENABLE_EXCHANGE_BOUNDARY_COMM=1 "
OPTS+="-DENABLE_INTERPOLATION_PL_COMM=0 "
OPTS+="-DENABLE_RESTRICTION_COMM=0 "
OPTS+="-DENABLE_RESTRICTION_ASYNC=0 "
OPTS+="-DENABLE_GPU_RDMA_TX=1 "
OPTS+="-DENABLE_GPU_RDMA_RX=1 "

# MPI/default stream options
#OPTS+="-DUSE_CUDA_AWARE_MPI "
#OPTS+="-DCUDA_API_PER_THREAD_DEFAULT_STREAM "
#OPTS+="-DUSE_DEFAULT_STREAM_MPI "
FLAGS_TESTS

OPTS+="-DPROFILE_NVTX_RANGES "
OPTS+="$GDSYNC_CPPFLAGS $CU_CPPFLAGS "
OPTS+="-DUSE_MPI_BARRIER "


# GSRB smoother (default)

#--NVCCFLAGS="-O0 -lineinfo $OPTS -Xptxas=-v --resource-usage --save-temps -lineinfo " \

./configure --CC=$CC --NVCC=$NVCC --CXX=$CXX \
    --CFLAGS="-O2 -fopenmp $OPTS " \
    --CXXFLAGS="-O2 $OPTS " \
    --NVCCFLAGS="-O2 -lineinfo $OPTS " \
    --CUDAARCH="$CUDA_ARCH " \
    --LDFLAGS="$CU_LDFLAGS $GDSYNC_LDFLAGS " \
    --LDLIBS="-lmp -lgdsync -lgdrapi -lcuda -libverbs " \
    --no-fe

:<<USAGE
usage: configure [-h] [--arch ARCH] [--petsc-dir PETSC_DIR]
                 [--petsc-arch PETSC_ARCH] [--with-hpm WITH_HPM] [--CC CC]
                 [--CXX CXX] [--CFLAGS CFLAGS] [--CXXFLAGS CXXFLAGS]
                 [--CPPFLAGS CPPFLAGS] [--NVCC NVCC] [--NVCCFLAGS NVCCFLAGS]
                 [--CUDAARCH CUDAARCH] [--LDFLAGS LDFLAGS] [--LDLIBS LDLIBS]
                 [--no-fe] [--no-fv] [--no-fv-mpi] [--fv-cycle {V,F,U}]
                 [--no-fv-subcomm]
                 [--fv-coarse-solver {bicgstab,cabicgstab,cg,cacg}]
                 [--fv-smoother {cheby,gsrb,jacobi,l1jacobi}]
USAGE

# Chebyshev smoother
#./configure --CC=$CC --NVCC=$NVCC --CFLAGS="-O2 -fopenmp $OPTS" --NVCCFLAGS="-O2 -lineinfo -lnvToolsExt $OPTS" --CUDAARCH="$CUDA_ARCH" --fv-smoother="cheby" --no-fe

make clean -C build
make -j3 -C build V=1
