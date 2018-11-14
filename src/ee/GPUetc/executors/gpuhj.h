#ifndef GPUHJ_H_
#define GPUHJ_H_

#include "types.h"
#include <cuda.h>
#include <sys/time.h>
#include <vector>
#include "gtable.h"
#include "gexpression.h"


namespace gpu {

class GPUHJ {
public:
	GPUHJ();

	GPUHJ(GTable outer_table,
			GTable inner_table,
			std::vector<ExpressionNode*> search_idx,
			ExpressionNode *end_expression,
			ExpressionNode *post_expression,
			ExpressionNode *initial_expression,
			ExpressionNode *skipNullExpr,
			ExpressionNode *prejoin_expression,
			ExpressionNode *where_expression,
			IndexLookupType lookup_type,
			int mSizeIndex);

	~GPUHJ();

	bool join();

	void getResult(RESULT *output) const;

	int getResultSize() const;

	void debug();

	static const uint64_t MAX_BUCKETS[];
private:
	GTable outer_table_, inner_table_;
	RESULT *join_result_;
	int result_size_;
	IndexLookupType lookup_type_;
	uint64_t maxNumberOfBuckets_;
	int m_sizeIndex_;

	GExpressionVector search_exp_;
	GExpression end_expression_;
	GExpression post_expression_;
	GExpression initial_expression_;
	GExpression skipNullExpr_;
	GExpression prejoin_expression_;
	GExpression where_expression_;

	/* For profiling */
	std::vector<unsigned long> index_hcount_, prefix_sum_, join_time_, rebalance_cost_, remove_empty_;
	unsigned long total_;

	void profiling();
	uint getPartitionSize() const;

	void IndexCount(ulong *index_count, ResBound *out_bound);
	void IndexCount(ulong *index_count, ResBound *out_bound, cudaStream_t stream);

	void HashJoinLegacy(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size);
	void HashJoinLegacy(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size, cudaStream_t stream);


	void decompose(RESULT *output, ResBound *in_bound, ulong *in_location, ulong *local_offset, int size);
	void decompose(RESULT *output, ResBound *in_bound, ulong *in_location, ulong *local_offset, int size, cudaStream_t stream);

	void Rebalance(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size);
	void Rebalance(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);
};

}
#endif
