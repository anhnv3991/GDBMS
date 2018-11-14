#ifndef GPUNIJ_H_
#define GPUNIJ_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/storage/gtable.h"
#include "GPUetc/expressions/gexpression.h"
#include <vector>
#include <string>


namespace voltdb {
class GpuSeqScan {
public:
public:
	GpuSeqScan();

	GpuSeqScan(GTable *output_table,
				GTable input_table,
				ExpressionNode *predicate,
				std::vector<ExpressionNode*> output_column_exp);

	~GpuSeqScan();

	bool execute();

	std::string debug() const;

private:
	GTable *output_table_, input_table_;
	GExpressionVector output_column_exp_;
	GExpression predicate_;

	//For profiling
	std::vector<unsigned long> allocation_, prejoin_, index_, expression_, ipsum_, epsum_, wtime_, joins_only_, rebalance_;
	struct timeval all_start_, all_end_;

	void profiling();

	uint getPartitionSize() const;

	unsigned long timeDiff(struct timeval start, struct timeval end);

	void seqScan();

};
}

#endif
