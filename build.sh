# find MPI compiler
CC=`which mpicc`
#CC=`which mpiicc`
CXX=`which mpic++`

# find NVCC compiler
NVCC=`which nvcc`

# set gpu architectures to compile for
CUDA_ARCH="-gencode code=sm_35,arch=compute_35 "
CUDA_ARCH="$CUDA_ARCH -gencode code=sm_60,arch=compute_60 "
#CUDA_ARCH="$CUDA_ARCH -gencode code=sm_70,arch=compute_70 "

# main tile size
OPTS="-DBLOCKCOPY_TILE_I=32 "
OPTS=" $OPTS -DBLOCKCOPY_TILE_J=4 "
OPTS=" $OPTS -DBLOCKCOPY_TILE_K=8 "

# special tile size for boundary conditions
OPTS=" $OPTS -DBOUNDARY_TILE_I=64 "
OPTS=" $OPTS -DBOUNDARY_TILE_J=16 "
OPTS=" $OPTS -DBOUNDARY_TILE_K=16 "

# max number of solves after warmup
OPTS=" $OPTS -DMAX_SOLVES=100 "

# host level threshold: number of grid elements
OPTS=" $OPTS -DHOST_LEVEL_SIZE_THRESHOLD=10000 "

# max number of solves after warmup
#OPTS=" $OPTS -DMAX_SOLVES=10 "

# unified memory allocation options
OPTS=" $OPTS -DCUDA_UM_ALLOC "
#cudaHostAlloc
OPTS=" $OPTS -DCUDA_UM_ZERO_COPY "

# MPI buffers allocation policy
#cudaHostAlloc
OPTS=" $OPTS -DMPI_ALLOC_ZERO_COPY "
#cudaMalloc
#OPTS=" $OPTS -DMPI_ALLOC_PINNED "

:<<AUTHOR_COMMENTS
OPTS=" $OPTS -DMPI_ALLOC_ZERO_COPY " - uses cudaMallocHost to allocate MPI buffers
OPTS=" $OPTS -DMPI_ALLOC_PINNED "  - uses cudaMalloc or malloc to allocate MPI buffers (depends on the level size)
If you comment out both options it will use cudaMallocManaged to allocate MPI buffers
AUTHOR_COMMENTS

# stencil optimizations
OPTS=" $OPTS -DUSE_REG "
OPTS=" $OPTS -DUSE_TEX "
#OPTS=" $OPTS -DUSE_SHM "

# GSRB smoother options
#OPTS=" $OPTS -DGSRB_FP "
#OPTS=" $OPTS -DGSRB_STRIDE2 "
#OPTS=" $OPTS -DGSRB_BRANCH "
#OPTS=" $OPTS -DGSRB_OOP "

# tools
#OPTS=" $OPTS -DUSE_PROFILE "
#OPTS=" $OPTS -DUSE_NVTX "
#OPTS=" $OPTS -DUSE_ERROR "

# override MVAPICH flags for C++
OPTS=" $OPTS -DMPICH_IGNORE_CXX_SEEK "
OPTS=" $OPTS -DMPICH_SKIP_MPICXX "  

#libmp flags
OPTS=" $OPTS -DENABLE_EXCHANGE_BOUNDARY_COMM=1 "
OPTS=" $OPTS -DENABLE_RESTRICTION_COMM=1 "
OPTS=" $OPTS -DENABLE_RESTRICTION_ASYNC=1 "
OPTS=" $OPTS -DENABLE_INTERPOLATION_COMM=1 "
OPTS=" $OPTS -DENABLE_INTERPOLATION_ASYNC=1 "

:<<FLAGS_TESTS
# MPI/default stream options
#OPTS=" $OPTS -DUSE_CUDA_AWARE_MPI "
#OPTS=" $OPTS -DCUDA_API_PER_THREAD_DEFAULT_STREAM "
#OPTS=" $OPTS -DUSE_DEFAULT_STREAM_MPI "
FLAGS_TESTS

OPTS=" $OPTS -DPROFILE_NVTX_RANGES "

MPI_INCLUDE=" -I$MPI_HOME/include "
MPI_LIB=" -L$MPI_HOME/lib -L/lib64 -L/lib "

CUDA_LIB=" -L$CUDA_HOME/lib64 -lcudart "
CUDA_INCLUDE=" -I$CUDA_HOME/include "

#GPUDirect Async required
[ -z "$PREFIX" ] && { PREFIX="$HOME/gdasync/Libraries"; }
GDASYNC_LIB="-L$PREFIX/lib"
GDASYNC_INCLUDE="-I$PREFIX/include"

OPTS=" $OPTS $CUDA_INCLUDE $MPI_INCLUDE $GDASYNC_INCLUDE "

CFLAGS="-O2 -fopenmp $OPTS"
CXXFLAGS="-O2 $OPTS"
NVCCFLAGS="-O2 -lineinfo $OPTS "
LDFLAGS="$CUDA_LIB $MPI_LIB $GDASYNC_LIB "
LDLIBS="-lmpcomm -lmp -lgdsync -lgdrapi -lcuda -libverbs "

# GSRB smoother (default)
set -x
./configure --CC=$CC --NVCC=$NVCC --CXX=$CXX --CFLAGS="$CFLAGS" --CXXFLAGS="$CXXFLAGS" --NVCCFLAGS="$NVCCFLAGS" --CUDAARCH="$CUDA_ARCH" --LDFLAGS="$LDFLAGS" --LDLIBS="$LDLIBS" --no-fe

# Chebyshev smoother: --fv-smoother="cheby"
make clean -C build
make -j3 -C build V=1
