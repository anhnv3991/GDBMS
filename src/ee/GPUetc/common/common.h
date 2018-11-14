#ifndef NODEDATA_H
#define NODEDATA_H

#include "common/types.h"
#include <cuda.h>

namespace gpu {

#define CUDAH __forceinline__ __host__ __device__
#define CUDAD __forceinline__ __device__

typedef struct {
	ValueType data_type;
} GColumnInfo;

inline void gassert(cudaError_t err_code, const char *file, int line)
{
	if (err_code != cudaSuccess) {
		fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(err_code), file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(err_code) gassert(err_code, __FILE__, __LINE__)


}

#endif
