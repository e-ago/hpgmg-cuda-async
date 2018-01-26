#!/usr/bin/env bash

if [[ $# -ne 6 ]]; then
    echo "Illegal parameters number: hpgmg-fv <MPI proc num> <use comm> <use async> <use gpu> <log box size> <num boxes>"
	exit 1
fi

NP=$1
if [[ $NP -lt 2 ]]; then
    echo "Illegal procs number: $NP"
	exit 1
fi

export PATH=$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

#Assuming OpenMPI
OMPI_params="$OMPI_params --mca btl openib,self"
OMPI_params="$OMPI_params --mca btl_openib_want_cuda_gdr 1"
OMPI_params="$OMPI_params --mca btl_openib_warn_default_gid_prefix 0"

set -x
$MPI_HOME/bin/mpirun -verbose  $OMPI_params   \
 -x GDS_CQ_MAP_SMART=0 -x GDS_ENABLE_DEBUG=0 -x MP_ENABLE_DEBUG=0 -x HPGMG_ENABLE_DEBUG=0 -x MP_EVENT_ASYNC=0 -x MP_ENABLE_WARN \
 -x LD_LIBRARY_PATH -x PATH \
 -x GDS_DISABLE_WRITE64=0 \
 -x GDS_SIMULATE_WRITE64=0 \
 -x GDS_DISABLE_INLINECOPY=0 \
 -x GDS_DISABLE_WEAK_CONSISTENCY=0 \
 -x GDS_DISABLE_MEMBAR=0 \
 -x OMP_NUM_THREADS=1 -x ASYNC_2_STREAMS=0 \
 -x COMM_USE_COMM=$2  -x COMM_USE_ASYNC=$3   -x COMM_USE_GPU_COMM=$4 \
 -x MP_DBREC_ON_GPU=0 -x MP_RX_CQ_ON_GPU=0 -x MP_TX_CQ_ON_GPU=0 \
 -x USE_MPI=1 \
 -x CUDA_PASCAL_FORCE_40_BIT=1 \
 -x GDS_FLUSHER_SERVICE=0 -x GDS_GPU_HAS_FLUSHER=0 \
 --map-by node -np $NP -hostfile hostfile ./wrapper.sh ./build/bin/hpgmg-fv $5 $6

echo "COMM_USE_COMM=$2"
echo "COMM_USE_ASYNC=$3"
echo "COMM_USE_GPU_COMM=$4"

# ./wrapper.sh  nvprof -o nvprof-kernel.%q{OMPI_COMM_WORLD_RANK}.nvprof ./build/bin/hpgmg-fv $2 $3

