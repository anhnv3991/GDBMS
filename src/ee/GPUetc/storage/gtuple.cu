#include <cuda.h>
#include <cuda_runtime.h>
#include "gtuple.h"
#include <sstream>

namespace gpu {

std::string Tuple::debug() const
{
	std::ostringstream output;

	int64_t *tuple_host = (int64_t*)malloc(sizeof(int64_t) * columns_);
	GColumnInfo *schema_host = (GColumnInfo*)malloc(sizeof(GColumnInfo) * columns_);

	checkCudaErrors(cudaMemcpy(tuple_host, tuple_, sizeof(int64_t) * columns_, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(schema_host, schema_, sizeof(GColumnInfo) * columns_, cudaMemcpyDeviceToHost));

	for (int i = 0; i < columns_; i++) {
		GNValue tmp(schema_host[i].data_type, tuple_host[i]);

		output << tmp.debug();
		if (i < columns_ - 1)
			output << "::";
	}

	output << std::endl;

	std::string retval(output.str());

	return retval;
}
}
