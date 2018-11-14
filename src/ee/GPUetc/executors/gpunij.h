/********************************
タプルの情報はここでまとめておく。

元のプログラムでは構造体のリストだったが、
GPUで動かすため配列のほうが向いていると思ったので
配列に変更している
********************************/

#ifndef GPUNIJ_H_
#define GPUNIJ_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/storage/gtable.h"
#include "GPUetc/expressions/gexpression.h"
#include <string>

namespace voltdb {

class GPUNIJ{
public:
	GPUNIJ();

	GPUNIJ(GTable outer_table,
			GTable inner_table,
			GTable *output,
			ExpressionNode *pre_join_predicate,
			ExpressionNode *join_predicate,
			ExpressionNode *where_predicate);

	~GPUNIJ();

	bool execute();

	std::string debug() const;

private:
	GTable outer_table_, inner_table_;
	GTable *output_;

	GExpression pre_join_predicate_;
	GExpression join_predicate_;
	GExpression where_predicate_;

	std::vector<unsigned long> allocation_, count_time_, scan_time_, join_time_, joins_only_;
	ulong all_time_;

	void profiling();

	void FirstEvaluation(ulong *first_count);
	void SecondEvaluation(RESULT *join_result, ulong *write_location);
	void Output(RESULT *join_result, int current_size);
};
}
#endif
