#ifndef KEY_INDEX_H_
#define KEY_INDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/common/common.h"

namespace gpu {
class GKeyIndex {
public:
	CUDAH GKeyIndex();
protected:
	int size_;
};

CUDAH GKeyIndex::GKeyIndex() {
	size_ = 0;
}
}

#endif
