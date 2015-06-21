#ifndef __ERROR_H__
#define __ERROR_H__

#include <stdio.h>
#include <errno.h>
#include <string.h>

extern int *_derrno;
extern int _herrno;

#define CUDA_ERROR_CHECK

#define CSAFECALL( err )     \
    if (cudaSuccess != err){ \
        fprintf(stderr, "[ERROR] CUDA error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    }

#define KERNCHECK( ... ) \
    do { \
        __VA_ARGS__; \
        cudaError err = cudaGetLastError(); \
        if (cudaSuccess != err){ \
            fprintf(stderr, "[ERROR] CUDA error (%i) at %s:%i : %s\n", err, __FILE__, __LINE__, cudaGetErrorString(err)); \
            if (err == 4) exit(-1); \
            goto error; \
        } \
\
        err = cudaDeviceSynchronize(); \
        if( cudaSuccess != err ) { \
            fprintf(stderr, "[ERROR] CUDA sync failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            goto error; \
        } \
\
    }   while (0);
        /*cudaMemcpy(&_herrno, _derrno, sizeof(int), cudaMemcpyDeviceToHost); \
        if (_herrno != 0){ \
            log_err("CUDA error inside of kernel"); \
            goto error; \
        }*/


#ifdef NDEBUG
#define debug(M, ...)
#else
#define debug(M, ...) \
    fprintf(stderr, "DEBUG %s:%d: " M "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#define clean_errno() \
    (errno == 0 ? "None" : strerror(errno))

#define log_err(M, ...) \
    fprintf(stderr, "[ERROR] (%s:%d: errno: %s) " M \
            "\n", __FILE__, __LINE__, clean_errno(), ##__VA_ARGS__)

#define log_warn(M, ...) \
    fprintf(stderr, "[WARN ] (%s:%d: errno: %s) " M \
            "\n", __FILE__, __LINE__, clean_errno(), ##__VA_ARGS__)

#define log_info(M, ...) \
    fprintf(stderr, "[INFO ] (%s:%d) " M "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define check(A, M, ...) if(!(A)) { log_err(M, ##__VA_ARGS__); errno=0; goto error; }
#define sentinel(M, ...)  { log_err(M, ##__VA_ARGS__); errno=0; goto error; }
#define check_mem(A) check((A), "Out of memory.")
#define check_debug(A, M, ...) if(!(A)) { debug(M, ##__VA_ARGS__); errno=0; goto error; }

#endif
