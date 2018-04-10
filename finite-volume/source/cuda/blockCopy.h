/*
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#define COPY_THREAD_BLOCK_SIZE			64
#define INCREMENT_THREAD_BLOCK_SIZE		64

#define READ(i)	__ldg(&read[i])

/* ++++++++++++++++++++++ EXCHANGE BOUNDARY ++++++++++++++++++++++ */

/* ============ Stream Functions ============ */

template<int log_dim, int block_type>
__global__ void copy_block_kernel(level_type level, int id, communicator_type exchange_ghosts)
{
  // one CUDA thread block operates on one HPGMG tile/block
  blockCopy_type block = exchange_ghosts.blocks[block_type][blockIdx.x];

  // copy 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block.dim.i;
  int   dim_j       = block.dim.j;
  int   dim_k       = block.dim.k;

  int  read_i       = block.read.i;
  int  read_j       = block.read.j;
  int  read_k       = block.read.k;
  int  read_jStride = block.read.jStride;
  int  read_kStride = block.read.kStride;

  int write_i       = block.write.i;
  int write_j       = block.write.j;
  int write_k       = block.write.k;
  int write_jStride = block.write.jStride;
  int write_kStride = block.write.kStride;

  double * __restrict__  read = block.read.ptr;
  double * __restrict__ write = block.write.ptr;
    
  int  read_box = block.read.box;
  int write_box = block.write.box;
  if(read_box >=0) 
     read = level.my_boxes[ read_box].vectors[id] + level.my_boxes[ read_box].ghosts*(1+level.my_boxes[ read_box].jStride+level.my_boxes[ read_box].kStride);
  if(write_box>=0)
    write = level.my_boxes[write_box].vectors[id] + level.my_boxes[write_box].ghosts*(1+level.my_boxes[write_box].jStride+level.my_boxes[write_box].kStride);

  for(int gid=threadIdx.x;gid<dim_i*dim_j*dim_k;gid+=blockDim.x){
    // simple linear mapping of 1D threads to 3D indices
    int k=(gid/dim_i)/dim_j;
    int j=(gid/dim_i)%dim_j;
    int i=gid%dim_i;

    int  read_ijk = (i+ read_i) + (j+ read_j)* read_jStride + (k+ read_k)* read_kStride;
    int write_ijk = (i+write_i) + (j+write_j)*write_jStride + (k+write_k)*write_kStride;
    write[write_ijk] = READ(read_ijk);
  }
}

template<int log_dim, int block_type>
__global__ void increment_block_kernel(level_type level, int id, double prescale, communicator_type exchange_ghosts)
{
  // one CUDA thread block operates on one HPGMG tile/block
  blockCopy_type block = exchange_ghosts.blocks[block_type][blockIdx.x];

  // copy 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block.dim.i;
  int   dim_j       = block.dim.j;
  int   dim_k       = block.dim.k;

  int  read_i       = block.read.i;
  int  read_j       = block.read.j;
  int  read_k       = block.read.k;
  int  read_jStride = block.read.jStride;
  int  read_kStride = block.read.kStride;

  int write_i       = block.write.i;
  int write_j       = block.write.j;
  int write_k       = block.write.k;
  int write_jStride = block.write.jStride;
  int write_kStride = block.write.kStride;

  double * __restrict__  read = block.read.ptr;
  double * __restrict__ write = block.write.ptr;

  if(block.read.box >=0){
     read = level.my_boxes[ block.read.box].vectors[id] + level.my_boxes[ block.read.box].ghosts*(1+level.my_boxes[ block.read.box].jStride+level.my_boxes[ block.read.box].kStride);
     read_jStride = level.my_boxes[block.read.box ].jStride;
     read_kStride = level.my_boxes[block.read.box ].kStride;
  }
  if(block.write.box>=0){
    write = level.my_boxes[block.write.box].vectors[id] + level.my_boxes[block.write.box].ghosts*(1+level.my_boxes[block.write.box].jStride+level.my_boxes[block.write.box].kStride);
    write_jStride = level.my_boxes[block.write.box].jStride;
    write_kStride = level.my_boxes[block.write.box].kStride;
  }

  for(int gid=threadIdx.x;gid<dim_i*dim_j*dim_k;gid+=blockDim.x){
    // simple linear mapping of 1D threads to 3D indices
    int k=(gid/dim_i)/dim_j;
    int j=(gid/dim_i)%dim_j;
    int i=gid%dim_i;

    int  read_ijk = (i+ read_i) + (j+ read_j)* read_jStride + (k+ read_k)* read_kStride;
    int write_ijk = (i+write_i) + (j+write_j)*write_jStride + (k+write_k)*write_kStride;
    write[write_ijk] = prescale*write[write_ijk] + READ(read_ijk);
  }
}
#undef  KERNEL
#define KERNEL(log_dim, block_type, stream) \
  copy_block_kernel<log_dim,block_type><<<grid,block,0,stream>>>(level,id,exchange_ghosts);

extern "C"
void cuda_copy_block(level_type level, int id, communicator_type exchange_ghosts, int block_type, cudaStream_t stream)
{
  int block = COPY_THREAD_BLOCK_SIZE;
  int grid = exchange_ghosts.num_blocks[block_type]; if(grid<=0) return;

  int log_dim = (int)log2((double)level.dim.i);
  switch(block_type){
    case 0: KERNEL_LEVEL_STREAM(log_dim,0,stream); CUDA_ERROR break;
    case 1: KERNEL_LEVEL_STREAM(log_dim,1,stream); CUDA_ERROR break;
    case 2: KERNEL_LEVEL_STREAM(log_dim,2,stream); CUDA_ERROR break;
    default: printf("CUDA ERROR: incorrect block type, %i\n", block_type);
  }
}

/* ++++++++++++++++++++++ INTERPOLATION ++++++++++++++++++++++ */
#undef  KERNEL
#define KERNEL(log_dim, block_type) \
  increment_block_kernel<log_dim,block_type><<<grid,block>>>(level,id,prescale,exchange_ghosts);

extern "C"
void cuda_increment_block(level_type level, int id, double prescale, communicator_type exchange_ghosts, int block_type)
{
  int block = INCREMENT_THREAD_BLOCK_SIZE;
  int grid = exchange_ghosts.num_blocks[block_type]; if(grid<=0) return;

  int log_dim = (int)log2((double)level.dim.i);
  switch(block_type){
    case 0: KERNEL_LEVEL(log_dim,0); CUDA_ERROR break;
    case 1: KERNEL_LEVEL(log_dim,1); CUDA_ERROR break;
    case 2: KERNEL_LEVEL(log_dim,2); CUDA_ERROR break;
    default: printf("CUDA ERROR: incorrect block type, %i\n", block_type);
  }
}


/* ============ Kernel Functions ============ */

template<int block_type>
__device__ void copy_block_fuse(level_type level, int id, communicator_type exchange_ghosts, int block_id, int thread_id, int block_dim)
{
    // one CUDA thread block operates on one HPGMG tile/block
  blockCopy_type block = exchange_ghosts.blocks[block_type][block_id];

  // copy 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block.dim.i;
  int   dim_j       = block.dim.j;
  int   dim_k       = block.dim.k;

  int  read_i       = block.read.i;
  int  read_j       = block.read.j;
  int  read_k       = block.read.k;
  int  read_jStride = block.read.jStride;
  int  read_kStride = block.read.kStride;

  int write_i       = block.write.i;
  int write_j       = block.write.j;
  int write_k       = block.write.k;
  int write_jStride = block.write.jStride;
  int write_kStride = block.write.kStride;

  double * __restrict__  read = block.read.ptr;
  double * __restrict__ write = block.write.ptr;
    
  int  read_box = block.read.box;
  int write_box = block.write.box;
  if(read_box >=0) 
     read = level.my_boxes[ read_box].vectors[id] + level.my_boxes[ read_box].ghosts*(1+level.my_boxes[ read_box].jStride+level.my_boxes[ read_box].kStride);
  if(write_box>=0)
    write = level.my_boxes[write_box].vectors[id] + level.my_boxes[write_box].ghosts*(1+level.my_boxes[write_box].jStride+level.my_boxes[write_box].kStride);

  for(int gid=threadIdx.x;gid<dim_i*dim_j*dim_k;gid+=blockDim.x){
    // simple linear mapping of 1D threads to 3D indices
    int k=(gid/dim_i)/dim_j;
    int j=(gid/dim_i)%dim_j;
    int i=gid%dim_i;

    int  read_ijk = (i+ read_i) + (j+ read_j)* read_jStride + (k+ read_k)* read_kStride;
    int write_ijk = (i+write_i) + (j+write_j)*write_jStride + (k+write_k)*write_kStride;
    write[write_ijk] = READ(read_ijk);
  }
}

//--------------------------

#include "cub/thread/thread_load.cuh"
#include "cub/thread/thread_store.cuh"
#include "../comm.h"
#include <mp/device.cuh>
using namespace cub;

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(V)                          \
    (*(volatile typeof (V) *)&(V))
#endif

//const int large_number = 1<<10;
#define TOT_SCHEDS 128
const int max_scheds = TOT_SCHEDS;
const int max_types = 3;

typedef struct sched_info {
//  mp::mlx5::gdsync::sem32_t sema;
  unsigned int block;
  unsigned int done[max_types];
} sched_info_t;

__device__ sched_info_t scheds[max_scheds];

__global__ void scheds_init()
{
  int j = threadIdx.x;
  assert(gridDim.x == 1);
  assert(blockDim.x >= max_scheds);
  if (j < max_scheds) {
 //   scheds[j].sema.sem = 0;
//    scheds[j].sema.value = 1;
    scheds[j].block = 0;
    for (int i = 0; i < max_types; ++i)
      scheds[j].done[i] = 0;
  }
}

__device__ static inline unsigned int elect_block(sched_info &sched)
{
  unsigned int ret;
  const int n_grids = gridDim.x; // BUG: account for .y and .z
  __shared__ unsigned int block;
  if (0 == threadIdx.x) {
    // 1st guy gets 0
    block = atomicInc(&sched.block, n_grids);
  }
  __syncthreads();
  ret = block;
  return ret;
}

__device__ static inline unsigned int elect_one(sched_info &sched, int grid_size, int idx)
{
  unsigned int ret;
  __shared__ unsigned int last;
  assert(idx < max_types);
  if (0 == threadIdx.x) {
    // 1st guy gets 0
    // ((old >= val) ? 0 : (old+1))
    last = atomicInc(&sched.done[idx], grid_size);

  }
  __syncthreads();
  ret = last;
  return ret;
}

typedef unsigned long long ns_t;

static __device__ ns_t getTimerNs()
{
        unsigned long long time = 0;
        asm("mov.u64  %0, %globaltimer;" : "=l"(time) );
        //time = clock();
        return (ns_t)time;
}

#if 0
#define DPROF_START(A) ns_t t_start = getTimerNs()
#define DPROF_END(A)   do { dprof.h[dprof.idx] = getTimerNs() - t_start; } while(0)
#else
#define DPROF_START(A) (void*)0
#define DPROF_END(A)   (void*)0
#endif

//#define TIMINGS_YES 1

//static __device__ ns_t bubu;
#ifdef TIMINGS_YES
  #define TIME_PACK 0
  #define TIME_LOCAL 1
  #define TIME_UNPACK 2
  #define TIME_SEND 3
  #define TIME_RECV 4
#endif

__global__ void fused_copy_block_kernel(level_type level, int id, communicator_type exchange_ghosts, int grid0, int grid1, int grid2, int max_grid01, int sched_id, struct comm_dev_descs *pdescs)
{
    assert(sched_id >= 0 && sched_id < max_scheds);
    assert(gridDim.x >= max_grid01+grid2+1);

    #ifdef TIMINGS_YES
    long long int start, stop;
    unsigned long long start_global, stop_global;
    #endif

    sched_info_t &sched = scheds[sched_id];
    int block = elect_block(sched);
    int index_send=0;
    
    //Force sequential sends
    int ordered_send=0;

    //First block wait
    if(block == 0)
    {
        #ifdef TIMINGS_YES
        if(threadIdx.x == 0)
        {
          start = clock64();
          start_global = getTimerNs();
        }
        #endif

        assert(blockDim.x >= pdescs->n_wait);
        if (threadIdx.x < pdescs->n_wait) {
            mp::device::mlx5::wait(pdescs->wait[threadIdx.x]);
            // write MP trk flag
            // note: no mem barrier here!!!
            mp::device::mlx5::signal(pdescs->wait[threadIdx.x]);
        }

        __syncthreads();

        #ifdef TIMINGS_YES
        if(threadIdx.x == 0)
        {
            stop_global = getTimerNs();
            stop = clock64();
            times[TIME_RECV] = exchange_ghosts.blocks[2][block].dim.i*exchange_ghosts.blocks[2][block].dim.j*exchange_ghosts.blocks[2][block].dim.k; //((double)(stop-start)*1000)/((double)875500);
            times_global[TIME_RECV] = (stop_global-start_global);
        }
        #endif

        if (0 == threadIdx.x) {
            //ThreadStore<STORE_CG>((int*)(&sched.done[1]), 1);
            // signal other blocks
            ACCESS_ONCE(sched.done[1]) = 1;
        }
    }
    else
    {
        block--;
        if (block < max_grid01)
        {
            // assign first N thread blocks to this task
            if (block < grid0)
            {
                #ifdef TIMINGS_YES
                if(0 == threadIdx.x && block == 0)
                {
                    start = clock64();
                    start_global = getTimerNs();
                }
                #endif
                // pack data
                copy_block_fuse<0>(level, id, exchange_ghosts, block, threadIdx.x, blockDim.x);

                #ifdef TIMINGS_YES
                if(0 == threadIdx.x && block == 0)
                {
                    stop_global = getTimerNs();
                    stop = clock64();
                    times[TIME_PACK] = exchange_ghosts.blocks[0][block].dim.i*exchange_ghosts.blocks[0][block].dim.j*exchange_ghosts.blocks[0][block].dim.k*blockDim.x; // ((double)(stop-start)*1000)/((double)875500);
                    times_global[TIME_PACK] = (stop_global-start_global);
                }
                #endif

                // elect last block to wait
                int last_block = elect_one(sched, grid0, 0); //__syncthreads(); inside
                if (0 == threadIdx.x) __threadfence();

                if (last_block == grid0-1) 
                {
                    #ifdef TIMINGS_YES
                        if(threadIdx.x == 0)
                        {
                            start = clock64();
                            start_global = getTimerNs();
                        }
                    #endif

                    if (threadIdx.x < pdescs->n_tx /* n_ready */) {
                        // wait for ready
                        gdsync::device::wait_geq(pdescs->ready[threadIdx.x]);
                        // signal NIC
                        if(ordered_send)
                        {
                            for(index_send=0; index_send < pdescs->n_tx; index_send++)
                                mp::device::mlx5::send(pdescs->tx[index_send]);
                            __syncthreads();
                        }
                        else
                            mp::device::mlx5::send(pdescs->tx[threadIdx.x]);
                    }

                    #ifdef TIMINGS_YES
                    if(threadIdx.x == pdescs->n_ready-1)
                    {
                        stop_global = getTimerNs();
                        stop = clock64();
                        times[TIME_SEND] = 0; //((double)(stop-start)*1000)/((double)875500);
                        times_global[TIME_SEND] = (stop_global-start_global);
                    }
                    #endif
                }
            }

            // maybe reuse same blocks for this task
            if (block < grid1) {
                #ifdef TIMINGS_YES
                if(0 == threadIdx.x && block == 0)
                {
                    start = clock64();
                    start_global = getTimerNs();
                }  
                #endif

                copy_block_fuse<1>(level, id, exchange_ghosts, block, threadIdx.x, blockDim.x);

                #ifdef TIMINGS_YES
                if(0 == threadIdx.x && block == 0)
                {
                    stop_global = getTimerNs();
                    stop = clock64();
                    times[TIME_LOCAL] = 0; //((double)(stop-start)*1000)/((double)875500);
                    times_global[TIME_LOCAL] = (stop_global-start_global);
                }
                #endif
            }
        }
        else 
        {
            // use other blocks to wait and unpack
            block -= max_grid01;
            //if (0 == threadIdx.x) printf("[%d][%d] id=%d unpack\n", pid, block, sched_id);
            if (0 <= block && block < grid2)
            {
                if (0 == threadIdx.x)
                {
                    while (ThreadLoad<LOAD_CG>(&sched.done[1]) < 1) { __threadfence_block(); }
                }

                __syncthreads();

                #ifdef TIMINGS_YES
                if(0 == threadIdx.x && block == 0){
                    start = clock64();
                    start_global = getTimerNs();
                }
                #endif
                // execute sub-task
                copy_block_fuse<2>(level, id, exchange_ghosts, block, threadIdx.x, blockDim.x);

                #ifdef TIMINGS_YES
                if(0 == threadIdx.x && block == 0)
                {
                    stop_global = getTimerNs();
                    stop = clock64();
                    times[TIME_UNPACK] = 0; //((double)(stop-start)*1000)/((double)875500);
                    times_global[TIME_UNPACK] = (stop_global-start_global);
                }
                #endif
            }
        }
  }
}

__global__ void fused_copy_block_kernel_inverted(level_type level, int id, communicator_type exchange_ghosts, int grid0, int grid1, int grid2, int max_grid01, int sched_id, struct comm_dev_descs *pdescs)
{
  assert(sched_id >= 0 && sched_id < max_scheds);
  assert(gridDim.x >= max_grid01+grid2+1);

  #ifdef TIMINGS_YES
    long long int start, stop;
    unsigned long long start_global, stop_global;
  #endif

  sched_info_t &sched = scheds[sched_id];
  int block = elect_block(sched);
  
  if (block < max_grid01)
  {
    // assign first N thread blocks to this task
    if (block < grid0)
    {
      // pack data
      copy_block_fuse<0>(level, id, exchange_ghosts, block, threadIdx.x, blockDim.x);
    
      // elect last block to wait
      int last_block = elect_one(sched, grid0, 0); //__syncthreads(); inside
      if (0 == threadIdx.x)
          __threadfence();

      if (last_block == grid0-1) 
      {
        if (threadIdx.x < pdescs->n_tx /* n_ready */) {
          // wait for ready
          gdsync::device::wait_geq(pdescs->ready[threadIdx.x]); 
          // signal NIC
          mp::device::mlx5::send(pdescs->tx[threadIdx.x]);
        }
      }
    }

    // maybe reuse same blocks for this task
    if (block < grid1) {
      copy_block_fuse<1>(level, id, exchange_ghosts, block, threadIdx.x, blockDim.x);
    }
  }
  else 
  {
    // use other blocks to wait and unpack
    block -= max_grid01;

    if(block == 0)
    {
      assert(blockDim.x >= pdescs->n_wait);
      if (threadIdx.x < pdescs->n_wait) {
        mp::device::mlx5::wait(pdescs->wait[threadIdx.x]);
        // write MP trk flag
        // note: no mem barrier here!!!
        mp::device::mlx5::signal(pdescs->wait[threadIdx.x]);
      }
      
      __syncthreads();

      if (0 == threadIdx.x) {
        // signal other blocks
        ACCESS_ONCE(sched.done[1]) = 1;   
      }
    }
    else
    {
      block--;
      if (0 <= block && block < grid2) {

          if (0 == threadIdx.x)
          {
            while (ThreadLoad<LOAD_CG>(&sched.done[1]) < 1) { __threadfence_block(); }
          }

          __syncthreads();
          
          // execute sub-task
          copy_block_fuse<2>(level, id, exchange_ghosts, block, threadIdx.x, blockDim.x);
        }
    }
    //if (0 == threadIdx.x) printf("[%d][%d] id=%d unpack\n", pid, block, sched_id); 
  }
}


static int n_scheds = TOT_SCHEDS;

extern "C"
void cuda_fused_copy_block(level_type level, int id, communicator_type exchange_ghosts, cudaStream_t stream, comm_dev_descs_t descs)
{
    int n_blocks = COPY_THREAD_BLOCK_SIZE;
    int max_grid01 = std::max(exchange_ghosts.num_blocks[0], exchange_ghosts.num_blocks[1]);
    int min_grids = std::min(std::min(exchange_ghosts.num_blocks[0], exchange_ghosts.num_blocks[1]), exchange_ghosts.num_blocks[2]);
    int fused_grid = max_grid01 + exchange_ghosts.num_blocks[2];
    
    assert(min_grids > 0);
    assert( descs->n_ready > 0 );

    if (n_scheds >= max_scheds) {
        scheds_init<<<1, max_scheds, 0, stream>>>();
        n_scheds = 0;
    }

    DBG("id=%d blocks=%d grids={%d,%d,%d} fused_grid:%d n_scheds: %d, descs: n_ready=%d n_tx=%d n_wait=%d\n", 
    id, n_blocks, exchange_ghosts.num_blocks[0], exchange_ghosts.num_blocks[1], exchange_ghosts.num_blocks[2], fused_grid, 
    n_scheds, descs->n_ready, descs->n_tx, descs->n_wait);

    #ifdef TIMINGS_YES
        double * times, * times_d;
        ns_t * times_global, * times_d_global;

        cudaHostAlloc( (void**)&times, 5*sizeof(double), cudaHostAllocMapped );
        cudaHostGetDevicePointer ( &times_d, times, 0 );
        times[0] = 0;
        times[1] = 0;
        times[2] = 0;
        times[3] = 0;
        times[4] = 0;

        cudaHostAlloc( (void**)&times_global, 5*sizeof(ns_t), cudaHostAllocMapped );
        cudaHostGetDevicePointer ( &times_d_global, times_global, 0 );
        times_global[0] = 0;
        times_global[1] = 0;
        times_global[2] = 0;
        times_global[3] = 0;
        times_global[4] = 0;

          //clockrate = 875500
        /*  cudaDeviceProp prop;
          cudaGetDeviceProperties(&prop, 0);
          cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate, sizeof(int), 0, cudaMemcpyHostToDevice);
        */
    #endif

    fused_copy_block_kernel<<<fused_grid+1, n_blocks, 0, stream>>>(level, id, exchange_ghosts, exchange_ghosts.num_blocks[0], exchange_ghosts.num_blocks[1], exchange_ghosts.num_blocks[2], max_grid01, n_scheds++, descs);

    #ifdef TIMINGS_YES
        cudaDeviceSynchronize();

        fprintf(stdout, "\n\n**** RANK %d TIMES, grid0: %d, grid1: %d, grid2:%d****\n", level.my_rank, exchange_ghosts.num_blocks[0], exchange_ghosts.num_blocks[1], exchange_ghosts.num_blocks[2]);
        fprintf(stdout, "RANK %d, pack: %f size, global: %d\n", level.my_rank, times[TIME_PACK], times_global[TIME_PACK]);
        fprintf(stdout, "RANK %d, local: %f size, global: %d\n", level.my_rank, times[TIME_LOCAL], times_global[TIME_LOCAL]);
        fprintf(stdout, "RANK %d, unpack: %f size, global: %d\n", level.my_rank, times[TIME_UNPACK], times_global[TIME_UNPACK]);
        fprintf(stdout, "RANK %d, send: %f size, global: %d\n", level.my_rank, times[TIME_SEND], times_global[TIME_SEND]);
        fprintf(stdout, "RANK %d, recv: %f size, global: %d\n", level.my_rank, times[TIME_RECV], times_global[TIME_RECV]);
        fprintf(stdout, "***************\n\n");
           
        cudaFreeHost(times);
        cudaFreeHost(times_global);
    #endif
}
