#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <error.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "gpunij.h"
#include "utilities.h"

#include <sstream>
#include <string>


namespace voltdb {

GPUNIJ::GPUNIJ()
{
		all_time_ = 0;
		output_ = NULL;
}

GPUNIJ::GPUNIJ(GTable outer_table,
				GTable inner_table,
				GTable *output,
				ExpressionNode *pre_join_predicate,
				ExpressionNode *join_predicate,
				ExpressionNode *where_predicate)
{
	/**** Table data *********/
	outer_table_ = outer_table;
	inner_table_ = inner_table;
	all_time_ = 0;
	output_ = output;

	/**** Expression data ****/
	pre_join_predicate_ = GExpression(pre_join_predicate);
	join_predicate_ = GExpression(join_predicate);
	where_predicate_ = GExpression(where_predicate);
}

GPUNIJ::~GPUNIJ()
{
	pre_join_predicate_.free();
	join_predicate_.free();
	where_predicate_.free();
}

bool GPUNIJ::execute(){

	struct timeval all_start, all_end;
	gettimeofday(&all_start, NULL);

	/******** Calculate size of blocks, grids, and GPU buffers *********/
	int count_size = outer_table_.getMaxTuplePerBlock();
	ulong jr_size = 0;
	RESULT *jresult_dev;
	ulong *count;

	/* Allocate GPU buffer for table data and counting data.
	 * The last element to store size of output result.
	 * The size of the output result is calculated by
	 * a prefix sum on count.
	 */
	checkCudaErrors(cudaMalloc(&count, (count_size + 1) * sizeof(ulong)));

	/******* Allocate GPU buffer for join condition *********/
	struct timeval cstart, cend, pcstart, pcend, jstart, jend, end_join;

	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (uint outer_idx = 0; outer_idx < outer_table_.getBlockNum(); outer_idx++) {
		outer_table_.moveToBlock(outer_idx);
		if (outer_table_.getCurrentRowNum() == 0)
			continue;
		for (uint inner_idx = 0; inner_idx < inner_table_.getBlockNum(); inner_idx++) {
			inner_table_.moveToBlock(inner_idx);

			gettimeofday(&cstart, NULL);
			FirstEvaluation(count);

			gettimeofday(&cend, NULL);

			gettimeofday(&pcstart, NULL);
			GUtilities::ExclusiveScan(count, outer_table_.getCurrentRowNum() + 1, &jr_size);
			gettimeofday(&pcend, NULL);

			count_time_.push_back(GUtilities::timeDiff(cstart, cend));
			scan_time_.push_back(GUtilities::timeDiff(pcstart, pcend));


			if (jr_size == 0) {
				gettimeofday(&end_join, NULL);
				joins_only_.push_back(GUtilities::timeDiff(cstart, end_join));
				continue;
			}

			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));
			gettimeofday(&jstart, NULL);
			SecondEvaluation(jresult_dev, count);
			gettimeofday(&jend, NULL);

			Output(jresult_dev, jr_size);

			join_time_.push_back((jend.tv_sec - jstart.tv_sec) * 1000000 + (jend.tv_usec - jstart.tv_usec));

			gettimeofday(&end_join, NULL);

			jr_size = 0;

			joins_only_.push_back(GUtilities::timeDiff(cstart, end_join));
			checkCudaErrors(cudaFree(jresult_dev));
		}
	}


	/******** Free GPU memory, unload module, end session **************/
	checkCudaErrors(cudaFree(count));
	gettimeofday(&all_end, NULL);

	all_time_ = GUtilities::timeDiff(all_start, all_end);
	return true;
}

void GPUNIJ::profiling()
{
	unsigned long allocation_time = 0, count_t = 0, join_t = 0, scan_t = 0, joins_only_time = 0;

	for (int i = 0; i < count_time_.size(); i++) {
		count_t += count_time_[i];
	}

	for (int i = 0; i < join_time_.size(); i++) {
		join_t += join_time_[i];
	}

	for (int i = 0; i < scan_time_.size(); i++) {
		scan_t += scan_time_[i];
	}

	for (int i = 0; i < joins_only_.size(); i++) {
		joins_only_time += joins_only_[i];
	}

	allocation_time = all_time_ - joins_only_time;
	printf("**********************************\n"
			"Allocation & data movement time: %lu\n"
			"count Time: %lu\n"
			"Prefix Sum Time: %lu\n"
			"Join Time: %lu\n"
			"Joins Only Time: %lu\n"
			"Total join time: %lu\n"
			"*******************************\n",
			allocation_time, count_t, scan_t, join_t, joins_only_time, all_time_);
}


extern "C" __global__ void firstEvaluation(GTable outer, GTable inner,
											int outer_rows, int inner_rows,
											ulong *pre_join_count,
											GExpression pre_join_pred, GExpression join_pred, GExpression where_pred,
											int64_t *val_stack, ValueType *type_stack)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GNValue res;
	int count = 0;
	GTuple outer_tuple, inner_tuple;

	for (int i = index; i < outer_rows; i += stride) {
		outer_tuple = outer.getGTuple(i);

		for (int j = 0; j < inner_rows; j++) {
			inner_tuple = inner.getGTuple(j);
			res = GNValue::getTrue();

			res = (pre_join_pred.getSize() > 0) ? pre_join_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride) : res;
			res = (res.isTrue() && join_pred.getSize() > 0) ? join_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride) : res;
			res = (res.isTrue() && where_pred.getSize() > 0) ? where_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride) : res;

			count += (res.isTrue()) ? 1 : 0;
		}
		__syncthreads();
	}

	if (index < outer_rows)
		pre_join_count[index] = count;
	if (index == 0)
		pre_join_count[outer_rows] = 0;
}

void GPUNIJ::FirstEvaluation(ulong *first_count)
{
	int outer_rows = outer_table_.getCurrentRowNum();
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));

	printf("start evaluate\n");
	firstEvaluation<<<grid_x, block_x>>>(outer_table_, inner_table_,
										outer_table_.getCurrentRowNum(), inner_table_.getCurrentRowNum(),
										first_count,
										pre_join_predicate_, join_predicate_, where_predicate_,
										val_stack, type_stack);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}

extern "C" __global__ void secondEvaluation(GTable outer, GTable inner,
											int outer_rows, int inner_rows,
											ulong *write_location, RESULT *output,
											GExpression pre_join_pred, GExpression join_pred, GExpression where_pred,
											int64_t *val_stack, ValueType *type_stack)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GNValue res;
	GTuple outer_tuple, inner_tuple;

	for (int i = index; i < outer_rows; i += stride) {
		int location = write_location[i];

		outer_tuple = outer.getGTuple(i);

		for (int j = 0; j < inner_rows; j++) {
			inner_tuple = inner.getGTuple(j);

			res = GNValue::getTrue();
			res = (pre_join_pred.getSize() > 0) ? pre_join_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride) : res;
			res = (res.isTrue() && join_pred.getSize() > 0) ? join_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride) : res;
			res = (res.isTrue() && where_pred.getSize() > 0) ?where_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride) : res;

			output[location].lkey = (res.isTrue()) ? i : (-1);
			output[location].rkey = (res.isTrue()) ? j : (-1);
			location++;
		}
	}
}

void GPUNIJ::SecondEvaluation(RESULT *join_result, ulong *write_location)
{
	int outer_rows = outer_table_.getCurrentRowNum();
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1) / block_x + 1;

	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));

	secondEvaluation<<<grid_x, block_x>>>(outer_table_, inner_table_,
											outer_table_.getCurrentRowNum(), inner_table_.getCurrentRowNum(),
											write_location, join_result,
											pre_join_predicate_, join_predicate_, where_predicate_,
											val_stack, type_stack);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}

extern "C" __global__ void outputResult(GTable outer, GTable inner, RESULT *join_result, GTable output, int starting_row, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTuple tuple;
	int outer_cols = outer.getColumnCount();
	int inner_cols = inner.getColumnCount();

	for (int i = index; i < size; i += stride) {
		if (join_result[i].lkey != -1 && join_result[i].rkey != -1) {
			tuple = outer.getGTuple(join_result[i].lkey);

			output.setGTuple(tuple, i + starting_row, 0, outer_cols);

			tuple = inner.getGTuple(join_result[i].rkey);

			output.setGTuple(tuple, i + starting_row, outer_cols, inner_cols);
		}
	}
}

void GPUNIJ::Output(RESULT *join_result, int current_size)
{
	int block_id, tuple_id;

	output_->getNextFreeTuple(&block_id, &tuple_id);

	output_->moveToBlock(block_id);
	int size = output_->getFreeTupleCount();
	size = (current_size < size) ? current_size : size;

	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	outputResult<<<grid_x, block_x>>>(outer_table_, inner_table_, join_result, *output_, 0, size);
	checkCudaErrors(cudaDeviceSynchronize());

	output_->setBlockRow(block_id, output_->getCurrentRowNum() + size);

	int current_tuple_count = output_->getTupleCount();
	current_tuple_count += size;
	output_->setTupleCount(current_tuple_count);
	current_size -= size;

	if (current_size <= 0)
		return;

	int tuples_per_block = output_->getMaxTuplePerBlock();

	int block_num = (current_size - 1) / tuples_per_block + 1;
	int starting_row = size;

	for (int i = 0; i < block_num; i++, starting_row += tuples_per_block) {
		tuples_per_block = (starting_row + tuples_per_block < current_size) ? tuples_per_block : current_size - starting_row + 1;
		block_x = (tuples_per_block <= BLOCK_SIZE_X) ? tuples_per_block : BLOCK_SIZE_X;
		grid_x = (tuples_per_block - 1) / block_x + 1;

		output_->addBlock();
		outputResult<<<grid_x, block_x>>>(outer_table_, inner_table_, join_result + starting_row, *output_, 0, tuples_per_block);
		checkCudaErrors(cudaDeviceSynchronize());
		output_->setBlockRow(block_id + i, tuples_per_block);
		current_tuple_count += tuples_per_block;
	}

	output_->setTupleCount(current_tuple_count);
}

std::string GPUNIJ::debug(void) const
{
	std::ostringstream output;

	output << "******** Debugging information ***********" << std::endl;

	output << "preJoinPredicate: " << std::endl;
	output << pre_join_predicate_.debug() << std::endl;

	output << "Join Predicate: " << std::endl;
	output << join_predicate_.debug() << std::endl;

	output << "Where Predicate: " << std::endl;
	output << where_predicate_.debug() << std::endl;

	output << "Outer Table: " << std::endl;
	output << outer_table_.debug() << std::endl;

	output << "Inner Table: " << std::endl;
	output << inner_table_.debug() << std::endl;

	std::string retval(output.str());

	return retval;
}

}


