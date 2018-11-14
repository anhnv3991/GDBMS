#ifndef GPROJECTION_
#define GPROJECTION_

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/storage/gtable.h"
#include "GPUetc/expressions/gexpression.h"


#include <iostream>
#include <string>
#include <vector>

namespace voltdb {
class GExecutorProjection {
public:
	GExecutorProjection();
	GExecutorProjection(GTable *output_table, GTable input_table, int *tuple_array, int *param_array, GNValue *param, std::vector<ExpressionNode *> expression);
	bool execute();
	std::string debug() const;

	~GExecutorProjection();
private:
	GTable input_;
	GTable *output_;

	int *tuple_array_;
	int *param_array_;
	GNValue *param_;

	GExpressionVector expression_;

	void evaluate();
};
}

#endif
