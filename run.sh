#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "Illegal parameters number: hpgmg-fv <MPI proc num> <use comm> <use async> <use gpu> <log dim> <param>" #"<EXCHANGE_HOST_ALLOC> <EXCHANGE_MALLOC> <HOST_LEVEL_SIZE_THRESHOLD> "
exit -1
fi

export PATH=$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

gdsSimWrite64=0
gdsDisableInlcpy=0
gdsDisableWeakConsistency=0
gdsDisableMembar=0
procNumber=$1
#var_MPI_ALLOC_ZERO_COPY=$7
#var_MPI_ALLOC_PINNED=$8
var_HOST_LEVEL_SIZE_THRESHOLD=10000
var_GDS_CQ_MAP_SMART=0 #16384
var_ASYNC_2_STREAMS=0 #$10

extra_params="$extra_params --mca btl openib,self"
extra_params="$extra_params --mca btl_openib_want_cuda_gdr 1"
extra_params="$extra_params --mca btl_openib_warn_default_gid_prefix 0"

$MPI_HOME/bin/mpirun -verbose  $extra_params   -x PATH    -x GDS_CQ_MAP_SMART=$var_GDS_CQ_MAP_SMART -x GDS_ENABLE_DEBUG=0    -x MP_ENABLE_DEBUG=0  \
-x MP_EVENT_ASYNC=0     -x MP_ENABLE_WARN     -x MP_GUARD_PROGRESS=0     -x CUDA_VISIBLE_DEVICES     -x LD_LIBRARY_PATH     \
-x SIZE     -x MAX_SIZE     -x KERNEL_TIME     -x CALC_SIZE     -x COMM_COMP_RATIO     -x USE_SINGLE_STREAM     -x USE_GPU_ASYNC \
-x COMM_USE_COMM=$2  -x COMM_USE_ASYNC=$3   -x COMM_USE_GPU_COMM=$4     -x COMM_USE_GDRDMA     -x HPGMG_ENABLE_DEBUG=0 \
-x GDS_DISABLE_WRITE64=0 \
-x GDS_SIMULATE_WRITE64=0 \
-x GDS_DISABLE_INLINECOPY=0 \
-x GDS_DISABLE_WEAK_CONSISTENCY=0 \
-x GDS_DISABLE_MEMBAR=0 \
-x OMP_NUM_THREADS=1 -x ASYNC_2_STREAMS=$var_ASYNC_2_STREAMS \
-x MP_DBREC_ON_GPU=0 -x MP_RX_CQ_ON_GPU=0 -x MP_TX_CQ_ON_GPU=0 \
-x USE_MPI=1 \
--map-by node  -np $procNumber -hostfile hostfile ./wrapper.sh ./build/bin/hpgmg-fv $5 $6

#-x MPI_ALLOC_ZERO_COPY=$var_MPI_ALLOC_ZERO_COPY -x MPI_ALLOC_PINNED=$var_MPI_ALLOC_PINNED -x HOST_LEVEL_SIZE_THRESHOLD=$var_HOST_LEVEL_SIZE_THRESHOLD \

echo "COMM_USE_COMM=$2"
echo "COMM_USE_ASYNC=$3"
echo "COMM_USE_GPU_COMM=$4"
#echo "MPI_ALLOC_ZERO_COPY=$7"
#echo "MPI_ALLOC_PINNED=$8"
#echo "HOST_LEVEL_SIZE_THRESHOLD=$9"

# ./wrapper.sh  nvprof -o nvprof-kernel.%q{OMPI_COMM_WORLD_RANK}.nvprof ./build/bin/hpgmg-fv $2 $3

