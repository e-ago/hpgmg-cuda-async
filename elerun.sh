#!/bin/bash

if [ "$#" -ne 9 ]; then
    echo "Illegal number of parameters: hpgmg-fv <MPI proc num> <use comm> <use async> <use gpu> <log dim> <param> <EXCHANGE_HOST_ALLOC> <EXCHANGE_MALLOC> <HOST_LEVEL_SIZE_THRESHOLD> "
exit -1
fi



export PATH=$PATH
#=/opt/openmpi/v1.10.0/bin:/usr/local/cuda-7.0/bin:/ivylogin/home/drossetti/bin:/ivylogin/home/drossetti/work/cuda_a/sw/misc/linux:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:.

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
#=$HOME/work/p4/cuda_a/sw/dev/gpu_drv/cuda_a/drivers/gpgpu/_out/Linux_amd64_release:/opt/openmpi/v1.10.0/lib64:/opt/openmpi/v1.10.0/lib:/usr/local/cuda-7.0/lib64:/usr/local/cuda-7.0/lib:/usr/local/cuda-7.0/jre/lib:/usr/local/cuda-7.0/extras/CUPTI/lib64:/usr/local/cuda-7.0/extras/CUPTI/lib:/usr/lib64/nvidia:/usr/local/cuda-5.5/lib64:$HOME/peersync/lib

gdsSimWrite64=0
gdsDisableInlcpy=1
gdsDisableWeakConsistency=1
gdsDisableMembar=1
procNumber=$1
var_MPI_ALLOC_ZERO_COPY=$7
var_MPI_ALLOC_PINNED=$8
var_HOST_LEVEL_SIZE_THRESHOLD=$9
var_GDS_CQ_WAIT_SMART=0 #16384
var_ASYNC_2_STREAMS=$10

mpirun -verbose     -x PATH    -x GDS_CQ_WAIT_SMART=$var_GDS_WAIT_SMART -x GDS_ENABLE_DEBUG=0    -x MP_ENABLE_DEBUG=0   -x MP_EVENT_ASYNC     -x MP_ENABLE_WARN     -x MP_GUARD_PROGRESS     -x CUDA_VISIBLE_DEVICES     -x LD_LIBRARY_PATH     -x SIZE     -x MAX_SIZE     -x KERNEL_TIME     -x CALC_SIZE     -x COMM_COMP_RATIO     -x USE_SINGLE_STREAM     -x USE_GPU_ASYNC \
-x COMM_USE_COMM=$2  -x COMM_USE_ASYNC=$3   -x COMM_USE_GPU_COMM=$4     -x COMM_USE_GDRDMA     -x HPGMG_ENABLE_DEBUG=0 \
-x MPI_ALLOC_ZERO_COPY=$var_MPI_ALLOC_ZERO_COPY -x MPI_ALLOC_PINNED=$var_MPI_ALLOC_PINNED -x HOST_LEVEL_SIZE_THRESHOLD=$var_HOST_LEVEL_SIZE_THRESHOLD \
-x MP_DBREC_ON_GPU=0 -x GDS_DISABLE_WRITE64=1 -x GDS_SIMULATE_WRITE64=$gdsSimWrite64 -x GDS_DISABLE_INLINECOPY=$gdsDisableInlcpy -x GDS_DISABLE_WEAK_CONSISTENCY=$gdsDisableWeakConsistency  -x GDS_DISABLE_MEMBAR=$gdsDisableMembar \
-x CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 -x OMP_NUM_THREADS=1 -x MV2_ENABLE_AFFINITY=0 \
-x CUDA_DISABLE_UNIFIED_MEMORY=0 -x ASYNC_2_STREAMS=$var_ASYNC_2_STREAMS \
--mca btl_openib_want_cuda_gdr 1 --map-by node     -np $procNumber  -mca btl_openib_warn_default_gid_prefix 0 ./wrapper.sh ./build/bin/hpgmg-fv $5 $6

echo "COMM_USE_COMM=$2"
echo "COMM_USE_ASYNC=$3"
echo "COMM_USE_GPU_COMM=$4"
echo "MPI_ALLOC_ZERO_COPY=$7"
echo "MPI_ALLOC_PINNED=$8"
echo "HOST_LEVEL_SIZE_THRESHOLD=$9"

#--mca btl_openib_want_cuda_gdr 1     --map-by node     -np $procNumber  -hostfile hostfile   -mca btl_openib_warn_default_gid_prefix 0 ./wrapper.sh ./build/bin/hpgmg-fv $2 $3


#--mca btl_openib_want_cuda_gdr 1     --map-by node     -np $procNumber  -hostfile hostfile   -mca btl_openib_warn_default_gid_prefix 0 ./wrapper.sh  nvprof -o nvprof-kernel.%q{OMPI_COMM_WORLD_RANK}.nvprof ./build/bin/hpgmg-fv $2 $3

