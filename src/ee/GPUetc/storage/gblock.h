#ifndef GBLOCK_H_
#define GBLOCK_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "GPUetc/common/common.h"
#include "gtuple.h"

namespace gpu {

class GBlock {
public:
	GBlock();
	GBlock(int columns);

	inline void setRowNumber(int row_num) {
		row_num_ = row_num;
	}

	inline void setColNumber(int col_num) {
		col_num_ = col_num;
	}

	bool addColumn();
	bool addRow();

	void addMultipleRows();

	CUDAH GTuple at(int row_idx);
	CUDAH GTuple operator[](int row_idx);

	~GBlock();
private:
	GColumnInfo *schema_;

	int64_t *primary_storage_;
	char *secondary_storage_;
	bool gpu_resident_;

	int row_num_;
	int col_num_;
	int offset_;
	int max_row_num_;
	int max_col_num_;

	GBlock *next, *prev;
};

CUDAH GTuple GBlock::at(int row_idx)
{
	return GTuple(primary_storage_ + row_idx, schema_, column_num_, stride_, secondary_storage_);
}

CUDAH GTuple GBlock::operator[](int row_idx)
{
	return GTuple(primary_storage_ + row_idx, schema_, column_num_, stride_, secondary_storage_);
}

}

#endif
