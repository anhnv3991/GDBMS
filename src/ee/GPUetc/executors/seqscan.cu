#include "seqscan.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <string>
#include <sstream>
#include "utilities.h"

namespace voltdb {
GpuSeqScan::GpuSeqScan()
{
	output_table_ = NULL;
}

GpuSeqScan::GpuSeqScan(GTable *output_table,
						GTable input_table,
						ExpressionNode *predicate,
						std::vector<ExpressionNode*> output_column_exp)
{
	output_table_ = output_table;
	input_table_ = input_table;
	predicate_ = GExpression(predicate);
	output_column_exp_ = GExpressionVector(output_column_exp);
}

GpuSeqScan::~GpuSeqScan()
{
	predicate_.free();
	output_column_exp_.free();
}

extern "C" __global__ void seqscanPredicateCheck(int *result, GTable input_table,
													int rows,
													GExpression predicate,
													int64_t *val_stack, ValueType *type_stack)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	GTuple in_tuple;
	GNValue res;

	for (int i = index; i < rows; i += stride) {
		in_tuple = input_table.getGTuple(i);

		res = (predicate.getSize() > 0) ? predicate.evaluate(&in_tuple, NULL, val_stack + index, type_stack + index, stride) : GNValue::getTrue();

		result[i] = (res.isTrue()) ? 1 : 0;
	}

	if (index == 0)
		result[index] = 0;
}

extern "C" __global__ void seqscanOutputResult(GTable output_table, GTable input_table, int *check_result, int *location, int rows,
												GExpressionVector output_column_exp,
												int64_t *val_stack, ValueType *type_stack)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	GNValue tmp;
	GTuple out_tuple, in_tuple;
	int exp_num = output_column_exp.size();

	for (int i = index; i < rows; i += stride) {
		if (check_result[i] == 1) {
			in_tuple = input_table.getGTuple(i);
			out_tuple = output_table.getGTuple(location[i]);

			for (int j = 0; j < exp_num; j++) {
				tmp = output_column_exp.at(j).evaluate(&in_tuple, NULL, val_stack + index, type_stack + index, stride);
				out_tuple.setGNValue(tmp, j);
			}
		}
		__syncthreads();
	}
}

void GpuSeqScan::seqScan()
{
	int rows = input_table_.getCurrentRowNum();
	int block_x = (rows < BLOCK_SIZE_X) ? rows : BLOCK_SIZE_X;

	int grid_x = (rows - 1) / block_x + 1;

	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));

	int *check_result, *location;

	checkCudaErrors(cudaMalloc(&check_result, sizeof(int) * rows));
	checkCudaErrors(cudaMalloc(&location, sizeof(int) * (rows + 1)));

	seqscanPredicateCheck<<<grid_x, block_x>>>(location, input_table_, rows, predicate_, val_stack, type_stack);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(check_result, location, sizeof(int) * rows, cudaMemcpyDeviceToDevice));

	GUtilities::ExclusiveScan<int>(location, rows + 1);

	seqscanOutputResult<<<grid_x, block_x>>>(*output_table_, input_table_, check_result, location, rows, output_column_exp_, val_stack, type_stack);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}

bool GpuSeqScan::execute()
{
	int block_num = input_table_.getBlockNum();
	int rows = 0;

	for (int i = 0; i < block_num; i++) {
		output_table_->addBlock();
		input_table_.moveToBlock(i);
		output_table_->setBlockRow(i, input_table_.getCurrentRowNum());
		rows += input_table_.getCurrentRowNum();
		output_table_->setTupleCount(rows);

		seqScan();
	}

	return true;
}



std::string GpuSeqScan::debug() const
{
	std::ostringstream output;

	output << "Input table: ";
	output << input_table_.debug();
	output << "End of input table" << std::endl;

	output << "Output table: ";
	output << output_table_->debug();
	output << "End of output table" << std::endl;

	output << "Predicate: ";
	output << predicate_.debug();
	output << "End of predicate" << std::endl;

	output << "Output Column Expression: ";
	output << output_column_exp_.debug() << std::endl;
	output << "End of output column expression" << std::endl;

	std::string retval(output.str());

	return retval;
}


}
