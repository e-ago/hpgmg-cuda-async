# alternatively set CUDA_VISIBLE_DEVICES appropriately, see README for details
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

# number of CPU threads executing coarse levels
export OMP_NUM_THREADS=4

# enable threads for MVAPICH
export MV2_ENABLE_AFFINITY=0

# Single GPU
./build/bin/hpgmg-fv 7 8

# MPI, one rank per GPU
#mpirun -np 2 ./build/bin/hpgmg-fv 7 8
