#include <cuda.h>
#include <cuda_runtime.h>
#include "gtable.h"
#include <iostream>
#include <string>
#include <sstream>

namespace gpu {

GTable::GTable() {
	name_ = NULL;
	block_list_host_ = NULL;
	schema_ = NULL;
	columns_ = 0;
	rows_ = 0;
	block_num_ = 0;
	indexes_ = NULL;
	index_num_ = 0;
	index_ = NULL;
}

GTable::GTable(char *name, int column_num)
{
	name_ = name;
	block_list_host_ = NULL;
	columns_ = column_num;
	rows_ = 0;
	block_num_ = 0;
	indexes_ = NULL;
	index_num_ = 0;
	index_ = NULL;

	checkCudaErrors(cudaMalloc(&schema_, sizeof(GColumnInfo) * column_num));
}

GTable::GTable(char *name, GColumnInfo *schema, int column_num)
{
	name_ = name;
	block_list_host_ = NULL;
	columns_ = column_num;
	rows_ = 0;
	block_num_ = 0;
	indexes_ = NULL;
	index_num_ = 0;
	index_ = NULL;

	checkCudaErrors(cudaMalloc(&schema_, sizeof(GColumnInfo) * column_num));
	checkCudaErrors(cudaMemcpy(schema_, schema, sizeof(GColumnInfo) * column_num, cudaMemcpyHostToDevice));
}

/*********************
 * Add a new block and
 * move to added block
 **********************/
void GTable::addBlock() {
	block_list_host_ = (GBlock*)realloc(block_list_host_, block_num_ + 1);

	checkCudaErrors(cudaMalloc(&block_list_host_[block_num_].data, MAX_BLOCK_SIZE_));
	block_list_host_[block_num_].columns = columns_;
	block_list_host_[block_num_].rows = 0;
	block_list_host_[block_num_].block_size = MAX_BLOCK_SIZE_;

	block_dev_ = block_list_host_[block_num_];
	index_ = indexes_;

	block_num_++;
}

void GTable::removeTable() {
	for (int i = 0; i < block_num_; i++) {
		checkCudaErrors(cudaFree(block_list_host_[i].data));
	}

	if (block_num_ > 0) {
		free(block_list_host_);
		block_list_host_ = NULL;
		block_num_ = 0;
	}

	for (int i = 0; i < index_num_; i++) {
		indexes_[i].removeIndex();
	}

	rows_ = 0;
}

void GTable::removeTableAll() {
	if (schema_ != NULL) {
		checkCudaErrors(cudaFree(schema_));
		schema_ = NULL;
	}

	for (int i = 0; i < block_num_; i++) {
		checkCudaErrors(cudaFree(block_list_host_[i].data));
	}

	if (block_num_ > 0) {
		free(block_list_host_);
		block_num_ = 0;
	}

	for (int i = 0; i < index_num_; i++) {
		indexes_[i].removeIndex();
	}

}

void GTable::removeBlock(int block_id) {
	if (block_id < block_num_) {
		checkCudaErrors(cudaFree(block_list_host_[block_id].data));
		memcpy(block_list_host_ + block_id, block_list_host_ + block_id + 1, sizeof(GBlock) * (block_num_ - block_id));
		free(block_list_host_ + block_num_ - 1);
	}
}

GTable::GBlock* GTable::getBlock(int blockId) {
	return block_list_host_ + blockId;
}

GTreeIndex *GTable::getCurrentIndex() {
	return index_;
}

int GTable::getBlockNum() {
	return block_num_;
}

int GTable::getIndexCount() {
	return index_num_;
}

char *GTable::getTableName() {
	return name_;
}

int GTable::getBlockTupleCount(int block_id) {
	assert(block_id < block_num_);

	return block_list_host_[block_id].rows;
}

bool GTable::isBlockFull(int block_id) {
	return (block_list_host_[block_id].rows >= block_list_host_[block_id].block_size/(columns_ * sizeof(int64_t)));
}

int GTable::getCurrentRowNum() const {
	return block_dev_.rows;
}

void GTable::deleteAllTuples()
{
	for (int i = 0; i < block_num_; i++) {
		checkCudaErrors(cudaFree(block_list_host_[i].data));
	}
	free(block_list_host_);
	block_num_ = 0;
	rows_ = 0;
}

void GTable::deleteTuple(int blockId, int tupleId)
{
	if (tupleId < 0 || tupleId > block_list_host_[blockId].rows) {
		printf("Error: tupleId out of range\n");
		return;
	}

	GBlock *target_block = block_list_host_ + blockId;
	int64_t *target_data = target_block->data;

	checkCudaErrors(cudaMemcpy(target_data + tupleId * columns_, target_data + (tupleId + 1) * columns_, (target_block->rows - tupleId) * columns_ * sizeof(int64_t), cudaMemcpyDeviceToDevice));
	target_block->rows -= 1;
}

void GTable::insertTuple(int64_t *tuple)
{
	int block_id, tuple_id;

	nextFreeTuple(&block_id, &tuple_id);

	int64_t *target_location = block_list_host_[block_id].data + tuple_id * columns_;

	checkCudaErrors(cudaMemcpy(target_location, tuple, columns_ * sizeof(int64_t), cudaMemcpyHostToDevice));
	block_list_host_[block_id].rows++;
	insertToAllIndexes(block_id, tuple_id);
	rows_++;
}

void GTable::insertToAllIndexes(int block_id, int tuple_id)
{
	for (int i = 0; i < index_num_; i++) {
		insertToIndex(block_id, tuple_id, i);
	}
}

void GTable::insertToIndex(int block_id, int tuple_id, int index_id)
{
	return;
}

/* INCOMPLETED */
void GTable::addIndex(int *key_idx, int key_size, GIndexType type)
{
	printf("Error: unsupported operation\n");
//	indexes_ = (GIndex*)realloc(indexes_, sizeof(GIndex) * (index_num_ + 1));
//	index_num_++;
}

void GTable::removeIndex()
{
	printf("Error: unsupported operation\n");
	exit(1);
}

void GTable::moveToBlock(int idx) {
	assert(idx < block_num_);
	block_dev_ = block_list_host_[idx];
	index_ = indexes_;
}

void GTable::nextFreeTuple(int *block_id, int *tuple_id)
{
	*block_id = -1;
	*tuple_id = -1;

	//First try to search for an available block
	for (int i = 0; i < block_num_; i++) {
		if (!isBlockFull(i)) {
			*block_id = i;
			*tuple_id = block_list_host_[i].rows;
			return;
		}
	}

	//All current blocks are full, allocate a new one
	GBlock new_block;

	checkCudaErrors(cudaMalloc(&new_block.data, MAX_BLOCK_SIZE_));
	new_block.columns = columns_;
	new_block.rows = 0;
	new_block.block_size = MAX_BLOCK_SIZE_;

	block_list_host_ = (GBlock*)realloc(block_list_host_, sizeof(GBlock) * (block_num_ + 1));
	block_list_host_[block_num_] = new_block;
	block_num_++;
	*block_id = block_num_ - 1;
	*tuple_id = 0;
}

std::string GTable::debug() const
{
	std::ostringstream output;

	output << "********** DEBUG FOR GPU **********" << std::endl;
	output << "Table name       : " << name_ << std::endl;
	output << "Number of columns: " << columns_ << std::endl;
	output << "Number of rows   : " << rows_ << std::endl;
	output << "Number of blocks : " << block_num_ << std::endl;
	output << "Number of indexes: " << index_num_ << std::endl;
	output << "Schema           : " << std::endl;

	GColumnInfo *schema = (GColumnInfo*)malloc(sizeof(GColumnInfo) * columns_);
	checkCudaErrors(cudaMemcpy(schema, schema_, sizeof(GColumnInfo) * columns_, cudaMemcpyDeviceToHost));

	for (int i = 0; i < columns_; i++) {
		output << "Column[" << i << "]:";

		switch (schema[i].data_type) {
		case VAL_INVALID: {
			output << "Invalid";
			break;
		}
		case VAL_NULL: {
			output << "Null";
			break;
		}
		case VAL_FOR_DIAGNOSTICS_ONLY_NUMERIC: {
			output << "Diagnostic";
			break;
		}
		case VAL_TINYINT: {
			output << "Tinyint";
			break;
		}
		case VAL_SMALLINT: {
			output << "Smallint";
			break;
		}
		case VAL_INTEGER: {
			output << "Integer";
			break;
		}
		case VAL_BIGINT: {
			output << "Bigint";
			break;
		}
		case VAL_DOUBLE: {
			output << "Double";
			break;
		}
		case VAL_VARCHAR: {
			output << "Varchar";
			break;
		}
		case VAL_TIMESTAMP: {
			output << "Timestamp";
			break;
		}
		case VAL_DECIMAL: {
			output << "Decimal";
			break;
		}
		case VAL_BOOLEAN: {
			output << "Boolean";
			break;
		}
		case VAL_ADDRESS: {
			output << "Address";
			break;
		}
		case VAL_VARBINARY: {
			output << "Varbinary";
			break;
		}
		case VAL_ARRAY: {
			output << "Array";
			break;
		}
		default: {
			output << "Invalid";
			break;
		}
		}

		if (i < columns_ - 1)
			output << " | ";
		else
			output << std::endl;
	}

	output << "Table data:" << std::endl;
	for (int i = 0; i < block_num_; i++) {
		int columns = block_list_host_[i].columns;
		int rows = block_list_host_[i].rows;

		if (rows * columns == 0)
			continue;
		int64_t *host_table = (int64_t*)malloc(sizeof(int64_t) * columns * rows);

		checkCudaErrors(cudaMemcpy(host_table, block_list_host_[i].data, sizeof(int64_t) * columns * rows, cudaMemcpyDeviceToHost));

		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < columns; k++) {
				GNValue tmp(schema[k].data_type, host_table[j * columns + k]);
				output << tmp.debug();

				if (k != columns - 1)
					output << "::";
			}
			output << std::endl;
		}

		free(host_table);
	}

	output << "********* END OF GPU DEBUG *********" << std::endl;

	std::string retval(output.str());

	return retval;
}
}
