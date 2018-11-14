#ifndef GEXPRESSION_H_
#define GEXPRESSION_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "gnvalue.h"
#include "gtuple.h"
#include <vector>

namespace gpu {

class GExpression {
public:
	typedef struct _TreeNode {
		ExpressionType type;	//type of
		int column_idx;			//Index of column in tuple, -1 if not tuple value
		int tuple_idx;			//0: left, outer, 1: right, inner
		GNValue value;			// Value of const, = NULL if not const
	} GTreeNode;

	typedef struct _ExpressionNode ExpressionNode;

	struct _ExpressionNode {
		ExpressionNode *left, *right;
		GTreeNode node;
	};


	CUDAH GExpression() {
		expression_ = NULL;
		size_ = 0;
	}

	/* Create a new expression, allocate the GPU memory for
	 * the expression and convert the input pointer-based
	 * tree expression to the desired expression form.
	 */
	GExpression(ExpressionNode *expression);

	/* Create a new expression from an existing GPU buffer. */
	CUDAH GExpression(GTreeNode *expression, int size) {
		expression_ = expression;
		size_ = size;
	}

	/* Create an expression from an input pointer-based tree expression */
	bool createExpression(ExpressionNode *expression);

	void free();

	CUDAD int getSize()
	{
		return size_;
	}

	CUDAD GNValue evaluate(int64_t *outer_tuple, int64_t *inner_tuple, GColumnInfo *outer_schema, GColumnInfo *inner_schema, GNValue *stack, int offset)
	{
		int top = 0;

		for (int i = 0; i < size_; i++) {
			GTreeNode tmp = expression_[i];

			switch (tmp.type) {
				case EXP_VALUE_TUPLE: {
					if (tmp.tuple_idx == 0) {
						stack[top] = GNValue(outer_schema[tmp.column_idx].data_type, outer_tuple[tmp.column_idx]);
						top += offset;
					} else if (tmp.tuple_idx == 1) {
						stack[top] = GNValue(inner_schema[tmp.column_idx].data_type, inner_tuple[tmp.column_idx]);
						top += offset;
					}
					break;
				}
				case EXP_VALUE_CONSTANT:
				case EXP_VALUE_PARAMETER: {
					stack[top] = tmp.value;
					top += offset;
					break;
				}
				case EXP_OP_AND: {
					stack[top - 2 * offset] = stack[top - 2 * offset] && stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_OR: {
					stack[top - 2 * offset] = stack[top - 2 * offset] || stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] == stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_NOT_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] != stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_LESS_THAN: {
					stack[top - 2 * offset] = stack[top - 2 * offset] < stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_LESS_THAN_OR_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] <= stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_GREATER_THAN: {
					stack[top - 2 * offset] = stack[top - 2 * offset] > stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_GREATER_THAN_OR_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] >= stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_ADD: {
					stack[top - 2 * offset] = stack[top - 2 * offset] + stack[top - offset];
					top -= offset;

					break;
				}
				case EXP_OP_SUBTRACT: {
					stack[top - 2 * offset] = stack[top - 2 * offset] - stack[top - offset];
					top -= offset;

					break;
				}
				case EXP_OP_DIVIDE: {
					stack[top - 2 * offset] = stack[top - 2 * offset] / stack[top - offset];
					top -= offset;

					break;
				}
				case EXP_OP_MULTIPLY: {
					stack[top - 2 * offset] = stack[top - 2 * offset] * stack[top - offset];
					top -= offset;

					break;
				}
				default: {
					return GNValue::getFalse();
				}
			}
		}

		return stack[0];
	}

	CUDAD GNValue evaluate(GTuple outer_tuple, GTuple inner_tuple, GNValue *stack, int offset)
	{
		int top = 0;

		for (int i = 0; i < size_; i++) {
			GTreeNode tmp = expression_[i];

			switch (tmp.type) {
				case EXP_VALUE_TUPLE: {
					if (tmp.tuple_idx == 0) {
						stack[top] = outer_tuple[tmp.column_idx];
						top += offset;
					} else if (tmp.tuple_idx == 1) {
						stack[top] = inner_tuple[tmp.column_idx];
						top += offset;
					}
					break;
				}
				case EXP_VALUE_CONSTANT:
				case EXP_VALUE_PARAMETER: {
					stack[top] = tmp.value;
					top += offset;
					break;
				}
				case EXP_OP_AND: {
					stack[top - 2 * offset] = stack[top - 2 * offset] && stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_OR: {
					stack[top - 2 * offset] = stack[top - 2 * offset] || stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] == stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_NOT_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] != stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_LESS_THAN: {
					stack[top - 2 * offset] = stack[top - 2 * offset] < stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_LESS_THAN_OR_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] <= stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_GREATER_THAN: {
					stack[top - 2 * offset] = stack[top - 2 * offset] > stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_GREATER_THAN_OR_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] >= stack[top - offset];
					top -= offset;
					break;
				}
				case EXP_OP_ADD: {
					stack[top - 2 * offset] = stack[top - 2 * offset] + stack[top - offset];
					top -= offset;

					break;
				}
				case EXP_OP_SUBTRACT: {
					stack[top - 2 * offset] = stack[top - 2 * offset] - stack[top - offset];
					top -= offset;

					break;
				}
				case EXP_OP_DIVIDE: {
					stack[top - 2 * offset] = stack[top - 2 * offset] / stack[top - offset];
					top -= offset;

					break;
				}
				case EXP_OP_MULTIPLY: {
					stack[top - 2 * offset] = stack[top - 2 * offset] * stack[top - offset];
					top -= offset;

					break;
				}
				default: {
					return GNValue::getFalse();
				}
			}
		}

		return stack[0];
	}


	static int getExpressionLength(ExpressionNode *expression);

	std::string debug() const;
private:


	bool buildPostExpression(GTreeNode *output_expression, ExpressionNode *expression, int *index);

	std::string printNode(GTreeNode node, int index) const;

	GTreeNode *expression_;
	int size_;
};

class GExpressionVector {
	using GExpression::GTreeNode;
	using GExpression::ExpressionNode;
public:
	CUDAH GExpressionVector();
	GExpressionVector(std::vector<ExpressionNode*> expression_list);
	CUDAH GExpressionVector(GTreeNode *expression_list, int *exp_size, int exp_num);

	CUDAH int size() const;
	CUDAH GExpression at(int exp_idx) const;
	CUDAH GExpression operator[](int exp_idx) const;

	void free();

	std::string debug() const;
private:
	GTreeNode *expression_;
	int *exp_size_;
	int exp_num_;
};

CUDAH GExpressionVector::GExpressionVector()
{
	expression_ = NULL;
	exp_size_ = NULL;
	exp_num_ = 0;
}

CUDAH GExpressionVector::GExpressionVector(GTreeNode *expression_list, int *exp_size, int exp_num)
{
	expression_ = expression_list;
	exp_size_ = exp_size;
	exp_num_ = exp_num;
}

CUDAH int GExpressionVector::size() const
{
	return exp_num_;
}

CUDAH GExpression GExpressionVector::at(int exp_idx) const
{
	if (exp_idx >= exp_num_)
		return GExpression();

	return GExpression(expression_ + exp_size_[exp_idx], exp_size_[exp_idx + 1] - exp_size_[exp_idx]);
}

CUDAH GExpression GExpressionVector::operator[](int exp_idx) const
{
	if (exp_idx >= exp_num_)
		return GExpression();

	return GExpression(expression_ + exp_size_[exp_idx], exp_size_[exp_idx + 1] - exp_size_[exp_idx]);
}

}

#endif
