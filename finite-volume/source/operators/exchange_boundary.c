//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
// perform a (intra-level) ghost zone exchange on vector id
//  NOTE exchange_boundary() only exchanges the boundary.  
//  It will not enforce any boundary conditions
//  BC's are either the responsibility of a separate function or should be fused into the stencil
// The argument shape indicates which of faces, edges, and corners on each box must be exchanged
//  If the specified shape exceeds the range of defined shapes, the code will default to STENCIL_SHAPE_BOX (i.e. exchange faces, edges, and corners)

#ifndef CUDACHECK
#define __CUDACHECK(stmt, cond_str)         \
    do {                \
        cudaError_t result = (stmt);                                    \
        if (cudaSuccess != result) {                                    \
          fprintf(stderr, "[%d] [%d] Assertion \"%s != cudaSuccess\" failed at %s:%d error=%d(%s)\n", \
                  getpid(), mpi_comm_rank, cond_str, __FILE__, __LINE__, result, cudaGetErrorString(result)); \
          exit(EXIT_FAILURE);                                           \
        }                                                               \
    } while (0)

#define CUDACHECK(stmt) __CUDACHECK(stmt, #stmt)
#endif


#include <assert.h>
#include "../debug.h"

void exchange_boundary_plain(level_type * level, int id, int shape)
{
	double _timeCommunicationStart = getTime();
	double _timeStart,getTime();

	// shape must be < STENCIL_MAX_SHAPES in order to safely index into exchange_ghosts[]
	if(shape>=STENCIL_MAX_SHAPES)shape=STENCIL_SHAPE_BOX;  
	int my_tag = (level->tag<<4) | shape;
	int buffer=0;
	int n;

	#ifdef USE_MPI
	int nMessages = level->exchange_ghosts[shape].num_recvs + level->exchange_ghosts[shape].num_sends;
	MPI_Request *recv_requests = level->exchange_ghosts[shape].requests;
	MPI_Request *send_requests = level->exchange_ghosts[shape].requests + level->exchange_ghosts[shape].num_recvs;

	// TODO: investigate why this is necessary for multi-GPU runs
	if(level->use_cuda && (level->num_ranks > 1))
	cudaDeviceSynchronize();

	// loop through packed list of MPI receives and prepost Irecv's...
	if(level->exchange_ghosts[shape].num_recvs>0)
	{
		_timeStart = getTime();

		PUSH_RANGE("irecv", WAIT_COL);

		#ifdef USE_MPI_THREAD_MULTIPLE
			#pragma omp parallel for schedule(dynamic,1)
		#endif
		for(n=0;n<level->exchange_ghosts[shape].num_recvs;n++){
			MPI_Irecv(level->exchange_ghosts[shape].recv_buffers[n],
				level->exchange_ghosts[shape].recv_sizes[n],
				MPI_DOUBLE,
				level->exchange_ghosts[shape].recv_ranks[n],
				my_tag,
				MPI_COMM_WORLD,
				&recv_requests[n]
			);
		}

		POP_RANGE;

		level->timers.ghostZone_recv += (getTime()-_timeStart);
	}

	// pack MPI send buffers...
	if(level->exchange_ghosts[shape].num_blocks[0])
	{
		_timeStart = getTime();

		PUSH_RANGE("pack", KERNEL_COL);

		if(level->use_cuda) {
			cuda_copy_block(*level,id,level->exchange_ghosts[shape],0,NULL);
			// synchronize so the CPU sees the updated buffers which will be used for MPI transfers
			cudaDeviceSynchronize();
		}
		else 
		{
			PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[shape].num_blocks[0])
			for(buffer=0;buffer<level->exchange_ghosts[shape].num_blocks[0];buffer++) {
				CopyBlock(level,id,&level->exchange_ghosts[shape].blocks[0][buffer]);
			}
		}

		POP_RANGE;
		level->timers.ghostZone_pack += (getTime()-_timeStart);
	}


	// loop through MPI send buffers and post Isend's...
	if(level->exchange_ghosts[shape].num_sends>0){
		PUSH_RANGE("send", SEND_COL);
		_timeStart = getTime();

		#ifdef USE_MPI_THREAD_MULTIPLE
			#pragma omp parallel for schedule(dynamic,1)
		#endif

		for(n=0;n<level->exchange_ghosts[shape].num_sends;n++){
			MPI_Isend(level->exchange_ghosts[shape].send_buffers[n],
				level->exchange_ghosts[shape].send_sizes[n],
				MPI_DOUBLE,
				level->exchange_ghosts[shape].send_ranks[n],
				my_tag,
				MPI_COMM_WORLD,
				&send_requests[n]
			); 
		}

		level->timers.ghostZone_send += (getTime()-_timeStart);
		POP_RANGE;
	}
	#endif


	// exchange locally... try and hide within Isend latency... 
	if(level->exchange_ghosts[shape].num_blocks[1])
	{
		PUSH_RANGE("local", KERNEL_COL);
		_timeStart = getTime();

		if (level->use_cuda) {
			cuda_copy_block(*level, id, level->exchange_ghosts[shape], 1,NULL);
		}
		else {
			PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[shape].num_blocks[1])
			for(buffer=0;buffer<level->exchange_ghosts[shape].num_blocks[1];buffer++){
				CopyBlock(level,id,&level->exchange_ghosts[shape].blocks[1][buffer]);
			}
		}

		level->timers.ghostZone_local += (getTime()-_timeStart);
		POP_RANGE;
	}

	// wait for MPI to finish...
	#ifdef USE_MPI 
	if(nMessages) {
		PUSH_RANGE("waitall", WAIT_COL);
		_timeStart = getTime();

		MPI_Waitall(nMessages,level->exchange_ghosts[shape].requests,level->exchange_ghosts[shape].status);
		#ifdef SYNC_DEVICE_AFTER_WAITALL
			cudaDeviceSynchronize();
		#endif

		POP_RANGE;
		level->timers.ghostZone_wait += (getTime()-_timeStart);
	}


	// unpack MPI receive buffers 
	if(level->exchange_ghosts[shape].num_blocks[2]){
		PUSH_RANGE("upack", KERNEL_COL);
		_timeStart = getTime();

		if(level->use_cuda) {
			cuda_copy_block(*level,id,level->exchange_ghosts[shape],2,NULL);
		}
		else {
			PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[shape].num_blocks[2])
			for(buffer=0;buffer<level->exchange_ghosts[shape].num_blocks[2];buffer++){
				CopyBlock(level,id,&level->exchange_ghosts[shape].blocks[2][buffer]);
			}
		}

		level->timers.ghostZone_unpack += (getTime()-_timeStart);
		POP_RANGE;
	}
	#endif

}

void exchange_boundary_comm(level_type * level, int id, int shape){
	double _timeStart;
	int buffer=0, n;
	int nMessages = level->exchange_ghosts[shape].num_recvs + level->exchange_ghosts[shape].num_sends;
	/*
	Even if comm_use_comm() == 1 and comm_use_model_sa() == 1, if level->use_cuda == 0 means that copy_block operations must be done by CPU.
	In this situation, use the async communication is useless because a cudaDeviceSynchronize() must be done before unpack
	*/

	DBG("NMsg=%d recvs=%d sends=%d\n", nMessages, level->exchange_ghosts[shape].num_recvs, level->exchange_ghosts[shape].num_sends);

	comm_request_t ready_requests[level->exchange_ghosts[shape].num_recvs];
	comm_request_t  recv_requests[level->exchange_ghosts[shape].num_recvs];
	comm_request_t  send_requests[level->exchange_ghosts[shape].num_sends];

	if(level->use_cuda && (level->num_ranks > 1)) cudaDeviceSynchronize();

	// loop through packed list of MPI receives and prepost Irecv's...
	if(level->exchange_ghosts[shape].num_recvs>0){
		PUSH_RANGE("irecv + send ready", SEND_COL);
		_timeStart = getTime();

		for(n=0;n<level->exchange_ghosts[shape].num_recvs;n++) {
			DBG("recv_rank[%d]=%d size=%d\n", n, 
			level->exchange_ghosts[shape].recv_ranks[n],
			level->exchange_ghosts[shape].recv_sizes[n]);

			comm_irecv(level->exchange_ghosts[shape].recv_buffers[n],
				level->exchange_ghosts[shape].recv_sizes[n],
				MPI_DOUBLE,
				&level->exchange_ghosts[shape].recv_buffers_reg[n],
				level->exchange_ghosts[shape].recv_ranks[n],
				&recv_requests[n]
			);

			comm_send_ready(level->exchange_ghosts[shape].recv_ranks[n], &ready_requests[n]);
		}

		level->timers.ghostZone_recv += (getTime()-_timeStart);
		POP_RANGE;
	}
	
	if(level->exchange_ghosts[shape].num_blocks[0] > 0) {
		// check if the device copy is ready 
		PUSH_RANGE("pack", KERNEL_COL);
		// pack MPI send buffers...
		_timeStart = getTime();
		
		if(level->use_cuda) {
			// launch kernel to pack MPI buffers
			cuda_copy_block(*level,id,level->exchange_ghosts[shape],0, NULL);
			// need to make sure the kernel completes before we submit MPI request
			cudaDeviceSynchronize();
		} 
		else {
			PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[shape].num_blocks[0])
			for(buffer=0;buffer<level->exchange_ghosts[shape].num_blocks[0];buffer++)
				{ CopyBlock(level,id,&level->exchange_ghosts[shape].blocks[0][buffer]); }
		}

		level->timers.ghostZone_pack += (getTime()-_timeStart);
		POP_RANGE;
	}

	int n_sends = level->exchange_ghosts[shape].num_sends;
	char send_msk[n_sends];

	// loop through MPI send buffers and post Isend's...
	if(level->exchange_ghosts[shape].num_sends > 0) {
		memset(send_msk, 0, sizeof(send_msk[0]) * n_sends);
		// initiate send, but don't wait for all ready msgs to come
		// resume later

		PUSH_RANGE("test ready + isend 1", SEND_COL);
		_timeStart = getTime();

		do {
			for(n=0; n<level->exchange_ghosts[shape].num_sends; n++)
			{
				if (!send_msk[n]) {
					int rdy = 0;
					comm_test_ready(level->exchange_ghosts[shape].send_ranks[n], &rdy);
					//comm_wait_ready(level->exchange_ghosts[shape].send_ranks[n]);
					//rdy = 1;
					if (rdy) {
						comm_isend(level->exchange_ghosts[shape].send_buffers[n],
						level->exchange_ghosts[shape].send_sizes[n],
						MPI_DOUBLE,
						&level->exchange_ghosts[shape].send_buffers_reg[n],
						level->exchange_ghosts[shape].send_ranks[n],
						&send_requests[n]);
						--n_sends;
						send_msk[n] = 1;

					}
				}
			}
		} while (0);

		level->timers.ghostZone_send += (getTime()-_timeStart);
		POP_RANGE;
	}

	// exchange locally... try and hide within Isend latency... 
	if(level->exchange_ghosts[shape].num_blocks[1])
	{
		PUSH_RANGE("local", KERNEL_COL);
		_timeStart = getTime();

		if (level->use_cuda) {
			cuda_copy_block(*level, id, level->exchange_ghosts[shape], 1, NULL);
		} else {
			PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[shape].num_blocks[1])
		
			for(buffer=0;buffer<level->exchange_ghosts[shape].num_blocks[1];buffer++)
				{ CopyBlock(level,id,&level->exchange_ghosts[shape].blocks[1][buffer]); }
		}

		level->timers.ghostZone_local += (getTime()-_timeStart);
		POP_RANGE;
	}

	if (nMessages)
	{
		if (level->exchange_ghosts[shape].num_sends)
		{
			PUSH_RANGE("test ready + isend 2", SEND_COL);
			_timeStart = getTime();
			// resume send here
			while (n_sends)
			{
				for(n=0;n<level->exchange_ghosts[shape].num_sends;n++)
				{
					if (!send_msk[n]) {
						int rdy = 0;
						comm_test_ready(level->exchange_ghosts[shape].send_ranks[n], &rdy);
						if (rdy) {
							comm_isend(level->exchange_ghosts[shape].send_buffers[n],
								level->exchange_ghosts[shape].send_sizes[n],
								MPI_DOUBLE,
								&level->exchange_ghosts[shape].send_buffers_reg[n],
								level->exchange_ghosts[shape].send_ranks[n],
								&send_requests[n]
							);
							--n_sends;
							send_msk[n] = 1;
						}
					}
				}
			};

			level->timers.ghostZone_send += (getTime()-_timeStart);
			POP_RANGE;
		}

		// wait for recv & sends
		PUSH_RANGE("wait", WAIT_COL);
		_timeStart = getTime();
		comm_flush();
		level->timers.ghostZone_wait += (getTime()-_timeStart);
		POP_RANGE;
	}

	// unpack MPI receive buffers 
	if(level->exchange_ghosts[shape].num_blocks[2])
	{
		PUSH_RANGE("upack", KERNEL_COL);
		_timeStart = getTime();

		if(level->use_cuda) {
			cuda_copy_block(*level,id,level->exchange_ghosts[shape],2, NULL);  
		} else {
			PRAGMA_THREAD_ACROSS_BLOCKS(level,buffer,level->exchange_ghosts[shape].num_blocks[2])
			for(buffer=0;buffer<level->exchange_ghosts[shape].num_blocks[2];buffer++){CopyBlock(level,id,&level->exchange_ghosts[shape].blocks[2][buffer]);}
		}

		level->timers.ghostZone_unpack += (getTime()-_timeStart);
		POP_RANGE;
	}
}

void exchange_boundary_async(level_type * level, int id, int shape){
	double _timeStart;
	int buffer=0, n;
	int nMessages = level->exchange_ghosts[shape].num_recvs + level->exchange_ghosts[shape].num_sends;
	cudaStream_t recvStream, sendStream;

	sendStream = level->stream;
	recvStream = level->stream_rec;
	/*
	Even if comm_use_comm() == 1 and comm_use_model_sa() == 1, if level->use_cuda == 0 means that copy_block operations must be done by CPU.
	In this situation, use the async communication is useless because a cudaDeviceSynchronize() must be done before unpack
	*/

	DBG("NMsg=%d recvs=%d sends=%d\n", nMessages, level->exchange_ghosts[shape].num_recvs, level->exchange_ghosts[shape].num_sends);

	assert( comm_use_model_sa() );
	comm_request_t ready_requests[level->exchange_ghosts[shape].num_recvs];
	comm_request_t  recv_requests[level->exchange_ghosts[shape].num_recvs];
	comm_request_t  send_requests[level->exchange_ghosts[shape].num_sends];

	// loop through packed list of MPI receives and prepost Irecv's...
	if(level->exchange_ghosts[shape].num_recvs>0)
	{
		PUSH_RANGE("send ready", SEND_COL);
		_timeStart = getTime();
		for(n=0;n<level->exchange_ghosts[shape].num_recvs;n++){
			DBG("recv_rank[%d]=%d size=%d\n", n, 
			level->exchange_ghosts[shape].recv_ranks[n],
			level->exchange_ghosts[shape].recv_sizes[n]);
			comm_irecv(level->exchange_ghosts[shape].recv_buffers[n],
						level->exchange_ghosts[shape].recv_sizes[n],
						MPI_DOUBLE,
						&level->exchange_ghosts[shape].recv_buffers_reg[n],
						level->exchange_ghosts[shape].recv_ranks[n],
						&recv_requests[n]
			);

			comm_send_ready_on_stream(level->exchange_ghosts[shape].recv_ranks[n], 
							&ready_requests[n], sendStream);
		}

		level->timers.ghostZone_recv += (getTime()-_timeStart);
		POP_RANGE;
	}

	// pack MPI send buffers...
	if(level->exchange_ghosts[shape].num_blocks[0] > 0) 
	{
		PUSH_RANGE("pack", KERNEL_COL);
		_timeStart = getTime();

		cuda_copy_block(*level,id,level->exchange_ghosts[shape],0, sendStream);

		level->timers.ghostZone_pack += (getTime()-_timeStart);
		POP_RANGE;
	}

	// loop through MPI send buffers and post Isend's...
	if(level->exchange_ghosts[shape].num_sends > 0)
	{	
		PUSH_RANGE("test ready + isend", SEND_COL);
		_timeStart = getTime();

		for(n=0;n<level->exchange_ghosts[shape].num_sends;n++)
		{
			DBG("send_rank[%d]=%d size=%d\n", n,
			level->exchange_ghosts[shape].send_ranks[n], 
			level->exchange_ghosts[shape].send_sizes[n]);

			comm_wait_ready_on_stream(level->exchange_ghosts[shape].send_ranks[n], sendStream);
			
			comm_isend_on_stream(level->exchange_ghosts[shape].send_buffers[n], 
								level->exchange_ghosts[shape].send_sizes[n],
								MPI_DOUBLE,
								&level->exchange_ghosts[shape].send_buffers_reg[n],
								level->exchange_ghosts[shape].send_ranks[n],
								&send_requests[n],
								sendStream
			);
		}

		POP_RANGE;
		
		level->timers.ghostZone_send += (getTime()-_timeStart);
	}

	// exchange locally... try and hide within Isend latency... 
	if(level->exchange_ghosts[shape].num_blocks[1])
	{
		PUSH_RANGE("local", KERNEL_COL);
		_timeStart = getTime();
		
		cuda_copy_block(*level, id, level->exchange_ghosts[shape], 1, sendStream);

		level->timers.ghostZone_local += (getTime()-_timeStart);
		POP_RANGE;
	}

	// wait for recv
	if (level->exchange_ghosts[shape].num_recvs) {
		PUSH_RANGE("wait recv", WAIT_COL);
		_timeStart = getTime();
		
		comm_wait_all_on_stream(level->exchange_ghosts[shape].num_recvs, recv_requests, recvStream);

		level->timers.ghostZone_wait += (getTime()-_timeStart);
		POP_RANGE;
	}

	// unpack MPI receive buffers 
	if(level->exchange_ghosts[shape].num_blocks[2])
	{
		PUSH_RANGE("upack", KERNEL_COL);
		_timeStart = getTime();

		cuda_copy_block(*level,id,level->exchange_ghosts[shape],2, recvStream);

		level->timers.ghostZone_unpack += (getTime()-_timeStart);
		POP_RANGE;
	}

	// wait for send
	if (level->exchange_ghosts[shape].num_sends > 0) {
		PUSH_RANGE("wait send", WAIT_COL);
		_timeStart = getTime();
	
		comm_wait_all_on_stream(level->exchange_ghosts[shape].num_sends, send_requests, sendStream);
	
		level->timers.ghostZone_wait += (getTime()-_timeStart);
		POP_RANGE;
	}

	if(level->exchange_ghosts[shape].num_recvs>0){
		PUSH_RANGE("wait ready", WAIT_COL);
		_timeStart = getTime();
		
		comm_wait_all_on_stream(level->exchange_ghosts[shape].num_recvs, ready_requests, sendStream);
		
		level->timers.ghostZone_wait += (getTime()-_timeStart);
		POP_RANGE;
	}  

	PUSH_RANGE("progress", KERNEL_COL);
	comm_progress();
	POP_RANGE;
}

void exchange_boundary_comm_fused_copy(level_type * level, int id, int shape){
	double _timeStart;
	int buffer=0, n=0;
	int nMessages = level->exchange_ghosts[shape].num_recvs + level->exchange_ghosts[shape].num_sends;
	comm_request_t ready_requests[level->exchange_ghosts[shape].num_recvs];
	comm_request_t recv_requests[level->exchange_ghosts[shape].num_recvs];
	comm_request_t send_requests[level->exchange_ghosts[shape].num_sends];

	assert(comm_use_model_sa());
	assert(level->use_cuda == 1);

	DBG("NMsg=%d recvs=%d sends=%d\n", nMessages, level->exchange_ghosts[shape].num_recvs, level->exchange_ghosts[shape].num_sends);

	// loop through packed list of MPI receives and prepost Irecv's...
	_timeStart = getTime();
	for(n=0;n<level->exchange_ghosts[shape].num_recvs;n++) {
		DBG("recv_rank[%d]=%d size=%d\n", n, 
			level->exchange_ghosts[shape].recv_ranks[n],
			level->exchange_ghosts[shape].recv_sizes[n]);

		COMM_CHECK(comm_irecv(level->exchange_ghosts[shape].recv_buffers[n],
			level->exchange_ghosts[shape].recv_sizes[n],
			MPI_DOUBLE,
			&level->exchange_ghosts[shape].recv_buffers_reg[n],
			level->exchange_ghosts[shape].recv_ranks[n],
			&recv_requests[n]
		));

		COMM_CHECK(comm_send_ready_on_stream(level->exchange_ghosts[shape].recv_ranks[n], &ready_requests[n], level->stream));
	}

	level->timers.ghostZone_recv += (getTime()-_timeStart);

	_timeStart = getTime();
	for(n=0;n<level->exchange_ghosts[shape].num_sends;n++){
		DBG("send_rank[%d]=%d size=%d\n", n,
			level->exchange_ghosts[shape].send_ranks[n], 
			level->exchange_ghosts[shape].send_sizes[n]);

		COMM_CHECK(comm_prepare_wait_ready(level->exchange_ghosts[shape].send_ranks[n]));
		COMM_CHECK(comm_prepare_isend(level->exchange_ghosts[shape].send_buffers[n],
			level->exchange_ghosts[shape].send_sizes[n],
			MPI_DOUBLE,
			&level->exchange_ghosts[shape].send_buffers_reg[n],
			level->exchange_ghosts[shape].send_ranks[n],
			&send_requests[n]
		));
	}
	level->timers.ghostZone_send += (getTime()-_timeStart);

	// prepare wait recv
	_timeStart = getTime();
	if (level->exchange_ghosts[shape].num_recvs) {
		COMM_CHECK(comm_prepare_wait_all(level->exchange_ghosts[shape].num_recvs, recv_requests));
	}
	level->timers.ghostZone_wait += (getTime()-_timeStart);

	// kernel
	_timeStart = getTime();
	
	comm_dev_descs_t descs = comm_prepared_requests();
	cuda_fused_copy_block(*level, id, level->exchange_ghosts[shape], level->stream, descs);
	
	level->timers.ghostZone_local += (getTime()-_timeStart);

	// wait for send
	if (level->exchange_ghosts[shape].num_sends) {
		PUSH_RANGE("wait send", WAIT_COL);
		_timeStart = getTime();
	
		comm_wait_all_on_stream(level->exchange_ghosts[shape].num_sends, send_requests, level->stream);
	
		level->timers.ghostZone_wait += (getTime()-_timeStart);
		POP_RANGE;
	}
	// wait for ready
	if(level->exchange_ghosts[shape].num_recvs){
		PUSH_RANGE("wait ready", WAIT_COL);
		_timeStart = getTime();

		comm_wait_all_on_stream(level->exchange_ghosts[shape].num_recvs, ready_requests, level->stream);

		level->timers.ghostZone_wait += (getTime()-_timeStart);
		POP_RANGE;
	}

	PUSH_RANGE("progress", KERNEL_COL);
	comm_progress();
	POP_RANGE;
}

void exchange_boundary(level_type * level, int id, int shape) {
	if(shape>=STENCIL_MAX_SHAPES)shape=STENCIL_SHAPE_BOX;  // shape must be < STENCIL_MAX_SHAPES in order to safely index into exchange_ghosts[]

	#ifdef STAT_TIMERS
	cudaDeviceSynchronize();
	#endif
	
	double _timeCommunicationStart = getTime();

	if (ENABLE_EXCHANGE_BOUNDARY_COMM && comm_use_comm())
	{
		communicator_type *ct = &level->exchange_ghosts[shape];
		if (comm_use_model_ki() && comm_use_model_sa() && level->use_cuda && ct->num_blocks[0] > 0  && ct->num_blocks[1] > 0 && ct->num_blocks[2] > 0)
		{
			PUSH_RANGE("exchange_kernel", OP_COL);
			exchange_boundary_comm_fused_copy(level, id, shape);
		}   
		else if(comm_use_model_sa() && level->use_cuda)
		{
			PUSH_RANGE("exchange_async", OP_COL);
			//comm_flush();
			exchange_boundary_async(level, id, shape);
		}
		else
		{
			PUSH_RANGE("exchange_comm", OP_COL);
			exchange_boundary_comm(level, id, shape);
		}
	}
	else {
		PUSH_RANGE("exchange_plain", OP_COL);
		exchange_boundary_plain(level, id, shape);
	}

	#ifdef STAT_TIMERS
	cudaDeviceSynchronize();
	#endif
	
	POP_RANGE;
	level->timers.ghostZone_total += (double)(getTime()-_timeCommunicationStart);
}
