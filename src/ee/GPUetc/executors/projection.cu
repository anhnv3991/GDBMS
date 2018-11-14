#include "projection.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include <sstream>

namespace voltdb {
GExecutorProjection::GExecutorProjection()
{
	output_ = NULL;
	tuple_array_ = NULL;
	param_array_ = NULL;
	param_ = NULL;
}

GExecutorProjection::GExecutorProjection(GTable *output_table, GTable input_table, int *tuple_array, int *param_array, GNValue *param, std::vector<ExpressionNode *> expression)
{
	output_ = output_table;
	input_ = input_table;

	int columns = output_->getColumnCount();

	if (tuple_array != NULL) {
		checkCudaErrors(cudaMalloc(&tuple_array_, sizeof(int) * columns));
		checkCudaErrors(cudaMemcpy(tuple_array_, tuple_array, sizeof(int) * columns, cudaMemcpyHostToDevice));
	} else
		tuple_array_ = NULL;

	if (param_array != NULL) {
		checkCudaErrors(cudaMalloc(&param_array_, sizeof(int) * columns));
		checkCudaErrors(cudaMemcpy(param_array_, param_array, sizeof(int) * columns, cudaMemcpyHostToDevice));
	} else
		param_array_ = NULL;

	if (param != NULL) {
		checkCudaErrors(cudaMalloc(&param_, sizeof(GNValue) * columns));
		checkCudaErrors(cudaMemcpy(param_, param, sizeof(GNValue) * columns, cudaMemcpyHostToDevice));
	} else
		param_ = NULL;

	expression_ = GExpressionVector(expression);
}

bool GExecutorProjection::execute()
{
	int rows;

	for (int i = 0; i < input_.getBlockNum(); i++) {
		input_.moveToBlock(i);
		output_->addBlock();
		rows = input_.getCurrentRowNum();
		output_->setBlockRow(i, rows);
		output_->setTupleCount(output_->getColumnCount() + rows);

		evaluate();
	}

	return true;
}

extern "C" __global__ void gevaluate0(GTable output, GTable input, int *tuple_array, int rows)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTuple input_tuple, output_tuple;
	GNValue tmp;
	int columns = output.getColumnCount();

	for (int i = index; i < rows; i += stride) {
		input_tuple = input.getGTuple(i);
		output_tuple = output.getGTuple(i);

		for (int j = 0; j < columns; j++) {
			tmp = input_tuple.getGNValue(tuple_array[j]);
			output_tuple.setGNValue(tmp, j);
		}
	}
}

extern "C" __global__ void gevaluate1(GTable output, int *param_array, GNValue *param, int rows)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTuple output_tuple;
	GNValue tmp;
	int columns = output.getColumnCount();

	for (int i = index; i < rows; i += stride) {
		output_tuple = output.getGTuple(i);

		for (int j = 0; j < columns; j++)
			output_tuple.setGNValue(param[param_array[j]], j);
	}
}

extern "C" __global__ void gevaluate2(GTable output, GTable input, GExpressionVector expression, int rows, int64_t *val_stack, ValueType *type_stack)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTuple input_tuple, output_tuple;
	GNValue tmp;
	int columns = output.getColumnCount();

	for (int i = index; i < rows; i += stride) {
		output_tuple = output.getGTuple(i);
		input_tuple = input.getGTuple(i);

		for (int j = 0; j < expression.size(); j++) {

			tmp = expression.at(j).evaluate(&input_tuple, NULL, val_stack + index, type_stack + index, stride);

			output_tuple.setGNValue(tmp, j);
		}
	}
}

void GExecutorProjection::evaluate()
{
	int rows = input_.getCurrentRowNum();
	int block_x = (rows > BLOCK_SIZE_X) ? BLOCK_SIZE_X : rows;
	int grid_x = (rows - 1) / block_x + 1;

	int64_t *val_stack = NULL;
	ValueType *type_stack = NULL;

	if (tuple_array_ != NULL) {
		gevaluate0<<<grid_x, block_x>>>(*output_, input_, tuple_array_, rows);
	} else if (param_array_ != NULL) {
		gevaluate1<<<grid_x, block_x>>>(*output_, param_array_, param_, rows);
	} else {
		checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * grid_x * block_x * MAX_STACK_SIZE));
		checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * grid_x * block_x * MAX_STACK_SIZE));

		gevaluate2<<<grid_x, block_x>>>(*output_, input_, expression_, rows, val_stack, type_stack);
	}
	checkCudaErrors(cudaDeviceSynchronize());

	if (val_stack != NULL)
		checkCudaErrors(cudaFree(val_stack));

	if (type_stack != NULL)
		checkCudaErrors(cudaFree(type_stack));
}

std::string GExecutorProjection::debug() const
{
	std::ostringstream output;

	output << "DEBUG Type: Projection Executor" << std::endl;
	output << "Input table:" << std::endl;
	output << input_.debug() << std::endl;
	output << "Output table:" << std::endl;
	output << output_->debug() << std::endl;
	output << "Tuple array:" << std::endl;

	int columns = input_.getColumnCount();

	if (columns > 0 && tuple_array_ != NULL) {
		for (int i = 0; i < columns; i++) {
			output << "Tuple: " << tuple_array_[i];
			if (i < columns - 1)
				output << "::";
		}
		output << std::endl;
	} else
		output << "Empty" << std::endl;

	output << "Param list:" << std::endl;
	if (columns > 0 && param_array_ != NULL && param_ != NULL) {

		for (int i = 0; i < columns; i++) {
			output << "[" << param_array_[i] << "]:" << param_[param_array_[i]].debug();
			if (i < columns - 1)
				output << "::";
		}
		output << std::endl;

	} else
		output << "Empty" << std::endl;

	output << expression_.debug() << std::endl;

	std::string retval(output.str());

	return retval;
}

GExecutorProjection::~GExecutorProjection()
{
	if (tuple_array_ != NULL) {
		checkCudaErrors(cudaFree(tuple_array_));
		tuple_array_ = NULL;
	}

	if (param_array_ != NULL) {
		checkCudaErrors(cudaFree(param_array_));
		param_array_ = NULL;
	}

	if (param_ != NULL) {
		checkCudaErrors(cudaFree(param_));
		param_ = NULL;
	}

	expression_.free();
}
}
