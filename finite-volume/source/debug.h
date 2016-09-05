#pragma once

#include <unistd.h>

extern int mpi_comm_rank;

#define STRDBG stderr

#ifdef __cplusplus
extern "C" {
#endif
int dbg_enabled();

#define DBG(FMT, ARGS...)                                               \
    do {                                                                \
        if (dbg_enabled()) {                                            \
            fprintf(STRDBG, "[%d] [%d] HPGMG %s(): " FMT,               \
                    getpid(), mpi_comm_rank, __FUNCTION__ , ## ARGS);   \
            fflush(STRDBG);                                             \
        }                                                               \
    } while(0)

#ifdef __cplusplus
}
#endif

#ifdef PROFILE_NVTX_RANGES
#include "nvToolsExt.h"

#define COMM_COL 1
#define SM_COL   2
#define SML_COL  3
#define OP_COL   4
#define COMP_COL 5
#define SOLVE_COL 6
#define WARMUP_COL 7
#define EXEC_COL 8

#define SEND_COL 9
#define WAIT_COL 10
#define KERNEL_COL 11


#define PUSH_RANGE(name,cid)																						\
	do {																																	\
	  const uint32_t colors[] = {																					\
            0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff, 0xff000000, 0xff0000ff, 0x55ff3300, 0xff660000, 0x66330000  \
		};																																	\
		const int num_colors = sizeof(colors)/sizeof(colors[0]);						\
		int color_id = cid%num_colors;																	\
    nvtxEventAttributes_t eventAttrib = {0};												\
    eventAttrib.version = NVTX_VERSION;															\
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;								\
    eventAttrib.colorType = NVTX_COLOR_ARGB;												\
    eventAttrib.color = colors[color_id];														\
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;							\
    eventAttrib.message.ascii = name;																\
    nvtxRangePushEx(&eventAttrib);																	\
	} while(0)

#define PUSH_RANGE_STR(cid, FMT, ARGS...)				\
	do {																					\
		char str[128];															\
		snprintf(str, sizeof(str), FMT, ## ARGS);		\
		PUSH_RANGE(str, cid);												\
	} while(0)


#define POP_RANGE do { nvtxRangePop(); } while(0)

#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif
