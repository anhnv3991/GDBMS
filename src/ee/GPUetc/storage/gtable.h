#ifndef GTABLE_H_
#define GTABLE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "gnvalue.h"
#include "common.h"
#include "macros.h"
#include "TreeIndex.h"
#include "HashIndex.h"
#include "KeyIndex.h"
#include "gblock.h"

namespace gpu {

class GTable {
	friend class GTuple;
	friend class GTreeIndex;
	friend class GHashIndex;

public:
	typedef enum {
		GTREE_INDEX_,
		GHASH_INDEX_,
	} GIndexType;

	/* Allocate an empty table */
	GTable();

	/* Allocate an empty table without schema. */
	GTable(char *name, int column_num);

	/* Allocate an empty table with schema. */
	GTable(char *name, GColumnInfo *schema, int column_num);

	/*****************************
	 * Host-side functions
	 *****************************/
	void addBlock();

	void removeBlock(int block_id);

	/****************************
	 * Only remove table data and
	 * set rows_ to zero.
	 * Keep schema. and columns_.
	 ****************************/
	void removeTable();

	/****************************
	 * Delete everything: table
	 * data and schema. Reset rows
	 * and columns_ to zero.
	 ***************************/
	void removeTableAll();

	GBlock *getBlock(int blockId);

	GTreeIndex *getCurrentIndex();

	int getBlockNum();

	int getIndexCount();

	char *getTableName();

	int getCurrentRowNum() const;

	void deleteAllTuples();


	void insertTuple(int64_t *tuple);

	void insertToAllIndexes(int blockId, int tupleId);

	void addIndex(int *key_idx, int key_size, GIndexType type);

	void removeIndex();

	void moveToBlock(int idx);

	int getMaxTuplePerBlock() {
		return MAX_BLOCK_SIZE/(columns_ * (int)(sizeof(int64_t)));
	}

	/********************************
	 * Block-level manipulation
	 ********************************/
	int getBlockTupleCount(int block_id);

	bool setBlockRow(int block_idx, int rows) {
		if (block_idx >= block_num_) {
			printf("Block does not exist\n");
			return false;
		}

		block_list_host_[block_idx].rows = rows;
		return true;
	}

	void setTupleCount(int rows) {
		rows_ = rows;
	}

	bool isBlockFull(int block_id);
	void deleteTuple(int blockId, int tupleId);
	void getNextFreeTuple(int *block_id, int *tuple_id) {
		nextFreeTuple(block_id, tuple_id);
	}

	int getFreeTupleCount() {
		return (block_dev_.block_size/(columns_ * (int)sizeof(int64_t)) - block_dev_.rows);
	}
	/********************************
	 * Device-side functions
	 *******************************/
	CUDAD GColumnInfo *getSchema() {
		return schema_;
	}

	CUDAD GBlock getBlock() {
		return block_dev_;
	}

	CUDAH int getColumnCount() const {
		return columns_;
	}

	CUDAH int getTupleCount() const {
		return rows_;
	}

	CUDAH GTuple getGTuple(int row) {
		return GTuple(block_dev_.data + columns_ * row, schema_, columns_);
	}

	/*****************************************
	 * Insert num elements to the tuple at row
	 * starting from start
	 *****************************************/
	CUDAD void setGTuple(GTuple tuple, int row, int start, int num) {
		int64_t *head = block_dev_.data + columns_ * row + start;

		for (int i = 0; i < num; i++)
			head[i] = tuple.tuple_[i];
	}

	std::string debug() const;

protected:
	GColumnInfo *schema_;
	GBlock block_dev_;
	GTreeIndex *index_;
	int columns_;
	int rows_;

private:
	void nextFreeTuple(int *blockId, int *tupleId);
	void insertToIndex(int block_id, int tuple_id, int index_id);

	char *name_;
	GBlock *block_list_host_;
	int block_num_;
	GTreeIndex *indexes_;
	int index_num_;
};
}

#endif
