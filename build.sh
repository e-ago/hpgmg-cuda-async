# find MPI compiler
CC=`which mpicc`
#CC=`which mpiicc`
CXX=`which mpic++`

# find NVCC compiler
NVCC=`which nvcc`

CUDA_ARCH=""
OPTS=""
# set gpu architectures to compile for
#CUDA_ARCH="$OPTS -gencode code=sm_35,arch=compute_35 "
#CUDA_ARCH="$OPTS -gencode code=sm_50,arch=compute_50 "
CUDA_ARCH="$CUDA_ARCH -gencode code=sm_60,arch=compute_60 "

# main tile size
OPTS="$OPTS -DBLOCKCOPY_TILE_I=32 "
OPTS="$OPTS -DBLOCKCOPY_TILE_J=4 "
OPTS="$OPTS -DBLOCKCOPY_TILE_K=8 "

# special tile size for boundary conditions
OPTS="$OPTS -DBOUNDARY_TILE_I=64 "
OPTS="$OPTS -DBOUNDARY_TILE_J=16 "
OPTS="$OPTS -DBOUNDARY_TILE_K=16 "

# host level threshold: number of grid elements
#OPTS="$OPTS -DHOST_LEVEL_SIZE_THRESHOLD=10000 "
#become environment variable

# max number of solves after warmup
OPTS="$OPTS -DMAX_SOLVES=100 "


# host level threshold: number of grid elements
OPTS="$OPTS -DHOST_LEVEL_SIZE_THRESHOLD=10000 "

# max number of solves after warmup
#OPTS="$OPTS -DMAX_SOLVES=10 "

# unified memory allocation options
OPTS="$OPTS -DCUDA_UM_ALLOC "
#cudaHostAlloc
OPTS="$OPTS -DCUDA_UM_ZERO_COPY "

# MPI buffers allocation policy
#cudaHostAlloc
OPTS="$OPTS -DMPI_ALLOC_ZERO_COPY "
#cudaMalloc
#OPTS="$OPTS -DMPI_ALLOC_PINNED "


:<<AUTHOR_COMMENTS
OPTS="$OPTS -DMPI_ALLOC_ZERO_COPY " - uses cudaMallocHost to allocate MPI buffers
OPTS="$OPTS -DMPI_ALLOC_PINNED "  - uses cudaMalloc or malloc to allocate MPI buffers (depends on the level size)
If you comment out both options it will use cudaMallocManaged to allocate MPI buffers
AUTHOR_COMMENTS

#With gpu direct async, unified memory cannot be used
#comment if you want to use MPI+Unified Memory
#OPTS="$OPTS -DCUDA_DISABLE_UNIFIED_MEMORY "

# stencil optimizations
OPTS="$OPTS -DUSE_REG "
OPTS="$OPTS -DUSE_TEX "
#OPTS="$OPTS -DUSE_SHM "

# GSRB smoother options
#OPTS="$OPTS -DGSRB_FP "
#OPTS="$OPTS -DGSRB_STRIDE2 "
#OPTS="$OPTS -DGSRB_BRANCH "
#OPTS="$OPTS -DGSRB_OOP "

# tools
#OPTS="$OPTS -DUSE_PROFILE "
#OPTS="$OPTS -DUSE_NVTX "
#OPTS="$OPTS -DUSE_ERROR "

# override MVAPICH flags for C++
OPTS="$OPTS -DMPICH_IGNORE_CXX_SEEK "
OPTS="$OPTS -DMPICH_SKIP_MPICXX "  

#libmp flags
OPTS="$OPTS -DENABLE_EXCHANGE_BOUNDARY_COMM=1 "
OPTS="$OPTS -DENABLE_RESTRICTION_COMM=1 "
OPTS="$OPTS -DENABLE_RESTRICTION_ASYNC=1 "
OPTS="$OPTS -DENABLE_INTERPOLATION_COMM=1 "
OPTS="$OPTS -DENABLE_INTERPOLATION_ASYNC=1 "

:<<FLAGS_TESTS
#BUG HERE: deadlock
OPTS="$OPTS -DENABLE_EXCHANGE_BOUNDARY_COMM=1 "
OPTS="$OPTS -DENABLE_INTERPOLATION_PL_COMM=0 "
OPTS="$OPTS -DENABLE_RESTRICTION_COMM=0 "
OPTS="$OPTS -DENABLE_RESTRICTION_ASYNC=0 "
OPTS="$OPTS -DENABLE_GPU_RDMA_TX=1 "
OPTS="$OPTS -DENABLE_GPU_RDMA_RX=1 "

# MPI/default stream options
#OPTS="$OPTS -DUSE_CUDA_AWARE_MPI "
#OPTS="$OPTS -DCUDA_API_PER_THREAD_DEFAULT_STREAM "
#OPTS="$OPTS -DUSE_DEFAULT_STREAM_MPI "
FLAGS_TESTS

OPTS="$OPTS -DPROFILE_NVTX_RANGES "
OPTS="$OPTS $GDSYNC_CPPFLAGS $CU_CPPFLAGS "
#OPTS="$OPTS -DUSE_MPI_BARRIER "
OPTS="$OPTS -DWAIT_READY "


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
