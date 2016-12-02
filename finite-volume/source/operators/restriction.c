//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
static inline void restriction_pc_block(level_type *level_c, int id_c, level_type *level_f, int id_f, blockCopy_type *block, int restrictionType){
  // restrict 3D array from read_i,j,k of read[] to write_i,j,k in write[] using piecewise constant restriction (cell averaged)
  int   dim_i       = block->dim.i; // calculate the dimensions of the resultant coarse block
  int   dim_j       = block->dim.j;
  int   dim_k       = block->dim.k;

  int  read_i       = block->read.i;
  int  read_j       = block->read.j;
  int  read_k       = block->read.k;
  int  read_jStride = block->read.jStride;
  int  read_kStride = block->read.kStride;

  int write_i       = block->write.i;
  int write_j       = block->write.j;
  int write_k       = block->write.k;
  int write_jStride = block->write.jStride;
  int write_kStride = block->write.kStride;

  double * __restrict__  read = block->read.ptr;
  double * __restrict__ write = block->write.ptr;
  if(block->read.box >=0){
     read_jStride = level_f->my_boxes[block->read.box ].jStride;
     read_kStride = level_f->my_boxes[block->read.box ].kStride;
     read = level_f->my_boxes[ block->read.box].vectors[id_f] + level_f->my_boxes[ block->read.box].ghosts*(1+level_f->my_boxes[ block->read.box].jStride+level_f->my_boxes[ block->read.box].kStride);
  }
  if(block->write.box>=0){
    write_jStride = level_c->my_boxes[block->write.box].jStride;
    write_kStride = level_c->my_boxes[block->write.box].kStride;
    write = level_c->my_boxes[block->write.box].vectors[id_c] + level_c->my_boxes[block->write.box].ghosts*(1+level_c->my_boxes[block->write.box].jStride+level_c->my_boxes[block->write.box].kStride);
  }



  int i,j,k;
  switch(restrictionType){
    case RESTRICT_CELL:
         for(k=0;k<dim_k;k++){
         for(j=0;j<dim_j;j++){
         for(i=0;i<dim_i;i++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk                            ]+read[read_ijk+1                          ] +
                                read[read_ijk  +read_jStride             ]+read[read_ijk+1+read_jStride             ] +
                                read[read_ijk               +read_kStride]+read[read_ijk+1             +read_kStride] +
                                read[read_ijk  +read_jStride+read_kStride]+read[read_ijk+1+read_jStride+read_kStride] ) * 0.125;
         }}}break;
    case RESTRICT_FACE_I:
         for(k=0;k<dim_k;k++){
         for(j=0;j<dim_j;j++){
         for(i=0;i<dim_i;i++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk                          ] +
                                read[read_ijk+read_jStride             ] +
                                read[read_ijk             +read_kStride] +
                                read[read_ijk+read_jStride+read_kStride] ) * 0.25;
         }}}break;
    case RESTRICT_FACE_J:
         for(k=0;k<dim_k;k++){
         for(j=0;j<dim_j;j++){
         for(i=0;i<dim_i;i++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk               ] +
                                read[read_ijk+1             ] +
                                read[read_ijk  +read_kStride] +
                                read[read_ijk+1+read_kStride] ) * 0.25;
         }}}break;
    case RESTRICT_FACE_K:
         for(k=0;k<dim_k;k++){
         for(j=0;j<dim_j;j++){
         for(i=0;i<dim_i;i++){
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk               ] +
                                read[read_ijk+1             ] +
                                read[read_ijk  +read_jStride] +
                                read[read_ijk+1+read_jStride] ) * 0.25;
         }}}break;
  }

}


//------------------------------------------------------------------------------------------------------------------------------
// perform a (inter-level) restriction on vector id_f of the fine level and stores the result in vector id_c on the coarse level
// restrictionType specifies whether this is either cell-averaged restriction, or one of three face-averaged restrictions
// piecewise constant restriction requires neither a ghost zone exchange nor a boundary condition
// This is a rather bulk synchronous implementation which packs all MPI buffers before initiating any sends
// Similarly, it waits for all remote data before copying any into local boxes.
// It does however attempt to overlap local restriction with MPI
void restriction_plain(level_type * level_c, int id_c, level_type *level_f, int id_f, int restrictionType){
  double _timeCommunicationStart = getTime();
  double _timeStart,_timeEnd;
  int buffer=0;
  int n;
  int my_tag = (level_f->tag<<4) | 0x5;

  #ifdef USE_MPI
  // by convention, level_f allocates a combined array of requests for both level_f sends and level_c recvs...
  int nMessages = level_c->restriction[restrictionType].num_recvs + level_f->restriction[restrictionType].num_sends;
  MPI_Request *recv_requests = level_f->restriction[restrictionType].requests;
  MPI_Request *send_requests = level_f->restriction[restrictionType].requests + level_c->restriction[restrictionType].num_recvs;

  // loop through packed list of MPI receives and prepost Irecv's...
  if(level_c->restriction[restrictionType].num_recvs>0){
    _timeStart = getTime();
    #ifdef USE_MPI_THREAD_MULTIPLE
    #pragma omp parallel for schedule(dynamic,1)
    #endif
    for(n=0;n<level_c->restriction[restrictionType].num_recvs;n++){
      MPI_Irecv(level_c->restriction[restrictionType].recv_buffers[n],
                level_c->restriction[restrictionType].recv_sizes[n],
                MPI_DOUBLE,
                level_c->restriction[restrictionType].recv_ranks[n],
                my_tag,
                MPI_COMM_WORLD,
                &recv_requests[n]
      );
    }
    _timeEnd = getTime();
    level_f->timers.restriction_recv += (_timeEnd-_timeStart);
  }


  // pack MPI send buffers...
  if(level_f->restriction[restrictionType].num_blocks[0]>0){
    _timeStart = getTime();
    if(level_f->use_cuda) {
      cuda_restriction(*level_c,id_c,*level_f,id_f,level_f->restriction[restrictionType],restrictionType,0);
      cudaDeviceSynchronize();  // switchover point: must synchronize GPU
    }
    else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->restriction[restrictionType].num_blocks[0])
      for(buffer=0;buffer<level_f->restriction[restrictionType].num_blocks[0];buffer++){
        restriction_pc_block(level_c,id_c,level_f,id_f,&level_f->restriction[restrictionType].blocks[0][buffer],restrictionType);
      }
    }
    _timeEnd = getTime();
    level_f->timers.restriction_pack += (_timeEnd-_timeStart);
  }

 
  // loop through MPI send buffers and post Isend's...
  if(level_f->restriction[restrictionType].num_sends>0){
    _timeStart = getTime();
    #ifdef USE_MPI_THREAD_MULTIPLE
    #pragma omp parallel for schedule(dynamic,1)
    #endif
    for(n=0;n<level_f->restriction[restrictionType].num_sends;n++){
      MPI_Isend(level_f->restriction[restrictionType].send_buffers[n],
                level_f->restriction[restrictionType].send_sizes[n],
                MPI_DOUBLE,
                level_f->restriction[restrictionType].send_ranks[n],
                my_tag,
                MPI_COMM_WORLD,
                &send_requests[n]
      );
    }
    _timeEnd = getTime();
    level_f->timers.restriction_send += (_timeEnd-_timeStart);
  }
  #endif


  // perform local restriction[restrictionType]... try and hide within Isend latency... 
  if(level_f->restriction[restrictionType].num_blocks[1]>0){
    _timeStart = getTime();
    if (level_f->use_cuda) {
      cuda_restriction(*level_c, id_c, *level_f, id_f, level_f->restriction[restrictionType], restrictionType, 1);
      if (!level_c->use_cuda) cudaDeviceSynchronize();  // switchover point: must synchronize GPU
    }
    else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->restriction[restrictionType].num_blocks[1])
      for(buffer=0;buffer<level_f->restriction[restrictionType].num_blocks[1];buffer++){
        restriction_pc_block(level_c,id_c,level_f,id_f,&level_f->restriction[restrictionType].blocks[1][buffer],restrictionType);
      }
    }
    _timeEnd = getTime();
    level_f->timers.restriction_local += (_timeEnd-_timeStart);
  }


  // wait for MPI to finish...
  #ifdef USE_MPI 
  if(nMessages){
    _timeStart = getTime();
    MPI_Waitall(nMessages,level_f->restriction[restrictionType].requests,level_f->restriction[restrictionType].status);
  #ifdef SYNC_DEVICE_AFTER_WAITALL
    cudaDeviceSynchronize();
  #endif
    _timeEnd = getTime();
    level_f->timers.restriction_wait += (_timeEnd-_timeStart);
  }



  // unpack MPI receive buffers 
  if(level_c->restriction[restrictionType].num_blocks[2]>0){
    _timeStart = getTime();
    if(level_c->use_cuda) {
      cuda_copy_block(*level_c,id_c,level_c->restriction[restrictionType],2, NULL);
    }
    else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->restriction[restrictionType].num_blocks[2])
      for(buffer=0;buffer<level_c->restriction[restrictionType].num_blocks[2];buffer++){
        CopyBlock(level_c,id_c,&level_c->restriction[restrictionType].blocks[2][buffer]);
      }
    }
    _timeEnd = getTime();
    level_f->timers.restriction_unpack += (_timeEnd-_timeStart);
  }
  #endif
 
 
  //level_f->timers.restriction_total += (double)(getTime()-_timeCommunicationStart);
}

#include <assert.h>
#include "../debug.h"

void restriction_comm(level_type * level_c, int id_c, level_type *level_f, int id_f, int restrictionType)
{
  double _timeCommunicationStart = getTime();
  double _timeStart,_timeEnd;
  int buffer=0;
  int n;  
  int use_async = level_c->use_cuda && level_f->use_cuda && comm_use_async() && ENABLE_RESTRICTION_ASYNC;

  // by convention, level_f allocates a combined array of requests for both level_f sends and level_c recvs...
  int nMessages = level_c->restriction[restrictionType].num_recvs + level_f->restriction[restrictionType].num_sends;
  comm_request_t  recv_requests[nMessages];
  comm_request_t  send_requests[nMessages];
  comm_request_t ready_requests[nMessages];


  DBG("id_c=%d type=%d nMessages=%d recvs=%d sends=%d\n", 
      id_c, restrictionType, nMessages, 
      level_c->restriction[restrictionType].num_recvs,
      level_f->restriction[restrictionType].num_sends);

  // loop through packed list of MPI receives and prepost Irecv's...
  if(level_c->restriction[restrictionType].num_recvs>0){
    _timeStart = getTime();
    for(n=0;n<level_c->restriction[restrictionType].num_recvs;n++){
      DBG("recv_ranks[%d]=%d type=%d n=%d\n", n, level_c->restriction[restrictionType].recv_ranks[n], restrictionType, n);

      comm_irecv(level_c->restriction[restrictionType].recv_buffers[n],
                 level_c->restriction[restrictionType].recv_sizes[n],
                 MPI_DOUBLE,
                 &level_f->restriction[restrictionType].recv_buffers_reg[n],
                 level_c->restriction[restrictionType].recv_ranks[n],
                 &recv_requests[n]);

      if (use_async) {
        comm_send_ready_on_stream(level_c->restriction[restrictionType].recv_ranks[n], 
                                  &ready_requests[n],
                                  level_c->stream);
      } else {
        comm_send_ready(level_c->restriction[restrictionType].recv_ranks[n], 
                        &ready_requests[n]);
      }
    }
    _timeEnd = getTime();
    level_f->timers.restriction_recv += (_timeEnd-_timeStart);
  }

  if(level_f->restriction[restrictionType].num_blocks[0]>0){
    _timeStart = getTime();
    if(level_f->use_cuda) {
      cuda_restriction(*level_c,id_c,*level_f,id_f,level_f->restriction[restrictionType],restrictionType,0);
      
      cudaDeviceSynchronize();  // switchover point: must synchronize GPU
PUSH_RANGE("Comm flush", COMM_COL);
//        cudaDeviceSynchronize();
        comm_flush();
POP_RANGE;
    }
    else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->restriction[restrictionType].num_blocks[0])
      for(buffer=0;buffer<level_f->restriction[restrictionType].num_blocks[0];buffer++){
        restriction_pc_block(level_c,id_c,level_f,id_f,&level_f->restriction[restrictionType].blocks[0][buffer],restrictionType);	
      }
    }
    _timeEnd = getTime();
    level_f->timers.restriction_pack += (_timeEnd-_timeStart);
  }

  // loop through MPI send buffers and post Isend's...
  if(level_f->restriction[restrictionType].num_sends>0){
    _timeStart = getTime();
    for(n=0;n<level_f->restriction[restrictionType].num_sends;n++){
      if (use_async) {
        comm_wait_ready_on_stream(level_f->restriction[restrictionType].send_ranks[n], level_f->stream);
        comm_isend_on_stream(level_f->restriction[restrictionType].send_buffers[n],
                             level_f->restriction[restrictionType].send_sizes[n],
                             MPI_DOUBLE,
                             &level_f->restriction[restrictionType].send_buffers_reg[n],
                             level_f->restriction[restrictionType].send_ranks[n],
                             &send_requests[n],
                             level_f->stream);
      } else {
        comm_wait_ready(level_f->restriction[restrictionType].send_ranks[n]);
        comm_isend(level_f->restriction[restrictionType].send_buffers[n],
                   level_f->restriction[restrictionType].send_sizes[n],
                   MPI_DOUBLE,
                   &level_f->restriction[restrictionType].send_buffers_reg[n],
                   level_f->restriction[restrictionType].send_ranks[n],
                   &send_requests[n]);
      }
    }
    _timeEnd = getTime();
    level_f->timers.restriction_send += (_timeEnd-_timeStart);
  }

  if(level_f->restriction[restrictionType].num_blocks[1]>0){
    _timeStart = getTime();
    if (level_f->use_cuda) {
      cuda_restriction(*level_c, id_c, *level_f, id_f, level_f->restriction[restrictionType], restrictionType, 1);
      if (!level_c->use_cuda) cudaDeviceSynchronize();  // switchover point: must synchronize GPU
    }
    else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_f->restriction[restrictionType].num_blocks[1])
      for(buffer=0;buffer<level_f->restriction[restrictionType].num_blocks[1];buffer++){
        restriction_pc_block(level_c,id_c,level_f,id_f,&level_f->restriction[restrictionType].blocks[1][buffer],restrictionType);
      }
    }
    _timeEnd = getTime();
    level_f->timers.restriction_local += (_timeEnd-_timeStart);
  }
  
   if (nMessages) {
    _timeStart = getTime();
     if (level_c->restriction[restrictionType].num_recvs) {
      DBG("wait on recv\n");
      if (use_async) {
        comm_wait_all_on_stream(level_c->restriction[restrictionType].num_recvs, 
                                recv_requests,
                                level_c->stream);
      }
      else
        comm_flush();
    }
    
    _timeEnd = getTime();
    level_f->timers.restriction_wait += (_timeEnd-_timeStart);
  }

  if(level_c->restriction[restrictionType].num_blocks[2]>0)
  {
    _timeStart = getTime();
    if(level_c->use_cuda) {
      cuda_copy_block(*level_c,id_c,level_c->restriction[restrictionType],2, NULL);
    }
    else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level_f,buffer,level_c->restriction[restrictionType].num_blocks[2])
      for(buffer=0;buffer<level_c->restriction[restrictionType].num_blocks[2];buffer++){
        CopyBlock(level_c,id_c,&level_c->restriction[restrictionType].blocks[2][buffer]);
	    }
    }
    _timeEnd = getTime();
    level_f->timers.restriction_unpack += (_timeEnd-_timeStart);
  }

  if (level_f->restriction[restrictionType].num_sends > 0 && use_async) {
    DBG("wait on send\n");
    _timeStart = getTime();
      comm_wait_all_on_stream(level_f->restriction[restrictionType].num_sends, 
                              send_requests,
                              level_c->stream);
    _timeEnd = getTime();
    level_f->timers.restriction_wait += (_timeEnd-_timeStart);
  }
   // level_f->timers.restriction_total += (uint64_t)(getTime()-_timeCommunicationStart);
}


// finer -> coarser
void restriction(level_type * level_c, int id_c, level_type *level_f, int id_f, int restrictionType)
{
  double _timeCommunicationStart = getTime();
  double timeFlush=0;
  if (ENABLE_RESTRICTION_COMM && comm_use_comm()) {
    // TODO enforce comm_flush()
    // if going down, CUDA->CPU
    // to get host memory updated
    /*
      f   c
      =====
      d   d
      d   h <- need flush
      h   h ??
      h   d IMPOSSIBLE currently
     */
    //if (level_f->use_cuda || level_c->use_cuda && comm_use_async()) {
/*
    //async
    int HOST_LEVEL_SIZE_THRESHOLD=10000; //default value
    const char *value = getenv("HOST_LEVEL_SIZE_THRESHOLD");
    if (value != NULL) {
        HOST_LEVEL_SIZE_THRESHOLD = atoi(value);
    }
      //In case of all levels use cuda
    if(HOST_LEVEL_SIZE_THRESHOLD > 0)
    {
*/      //useless !level_c->use_cuda ??
      
#if 0
      if (!level_c->use_cuda || !level_f->use_cuda && comm_use_async()) {
        PUSH_RANGE("Comm flush", COMM_COL);
        cudaDeviceSynchronize();
        comm_flush();
        POP_RANGE;
      } 
#endif

/*    }
    else
      comm_progress();
*/
    PUSH_RANGE("restriction_comm", OP_COL);
    restriction_comm(level_c, id_c, level_f, id_f, restrictionType);

  } else {
    PUSH_RANGE("restriction_plain", OP_COL);
    restriction_plain(level_c, id_c, level_f, id_f, restrictionType);
  }

  level_f->timers.restriction_total += (double)(getTime()-_timeCommunicationStart);

  POP_RANGE;
}
