#include "stdlib.h"
#include "stdio.h"
#include "debug.h"

int mpi_comm_rank = 0;

int dbg_enabled()
{
    static int dbg_is_enabled = -1;
    if (-1 == dbg_is_enabled) {        
        const char *env = getenv("HPGMG_ENABLE_DEBUG");
        if (env) {
            int en = atoi(env);
            dbg_is_enabled = !!en;
            printf("HPGMG_ENABLE_DEBUG=%s\n", env);
        } else
            dbg_is_enabled = 0;
    }
    return dbg_is_enabled;
}
