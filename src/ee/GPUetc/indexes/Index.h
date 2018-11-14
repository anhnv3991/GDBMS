#ifndef GINDEX_H_
#define GINDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "gtuple.h"
#include "common.h"

namespace gpu {

class GIndex {
public:
	CUDAH GIndex();

	virtual void addEntry(GTuple new_tuple) = 0;

	virtual void addBatchEntry(int64_t *table, GColumnInfo *schema, int rows, int columns) = 0;

	virtual void merge(int old_left, int old_right, int new_left, int new_right) = 0;

	virtual void removeIndex() = 0;

	virtual ~GIndex();
};

}

#endif
