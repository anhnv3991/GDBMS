#include "gblock.h"

namespace gpu {

GBlock::GBlock()
{
	//Currently, maximum size of a block is 64MB
	checkCudaErrors(cudaMalloc(&primary_storage_, MAX_BLOCK_SIZE * sizeof(int64_t)));
	secondary_storage_ = NULL;
	gpu_resident_ = true;

	row_num_ = 0;
	col_num_ = 0;
	max_row_num_ = MAX_ROWS_PER_BLOCK;
	max_col_num_ = MAX_COLS_PER_BLOCK;
	next = prev = NULL;
	offset_ = max_row_num_;
	schema_ = NULL;
}

__global__ void copyTable(int64_t *source, int64_t *destination, int row_num, int col_num, int sstride, int dstride, int sbase_id, int dbase_id)
{
	int row_id = threadIdx.x + blockIdx.x * blockDim.x;
	int col_id = blockIdx.y;

	if (row_id < row_num && col_id < col_num)
		destination[col_id * dstride + row_id + dbase_id] = source[col_id * sstride + row_id + sbase_id];
}

void copyTableHost(int64_t *source, int64_t *destination, int row_num, int col_num, int source_stride, int destination_stride, int sbase_id, int dbase_id)
{
	for (int col = 0; col < col_num; col++) {
		for (int row = sbase_id; row < row_num; row++)
	}
}

bool GBlock::addColumn()
{
	if (gpu_resident_) {
		if (col_num_ == max_col_num_) {
			int new_max_row_num, new_max_col_num;

			new_max_col_num = max_col_num_ + ADDITIONAL_COLS;
			new_max_row_num = MAX_ROWS_PER_BLOCK / new_max_col_num;

			// One new block for the upper half
			GBlock *new_block = new GBlock;

			new_block->max_col_num_ = new_max_col_num;
			new_block->max_row_num_ = new_max_row_num;
			new_block->offset_ = new_max_row_num;

			// Replace the current block with the new block
			if (prev != NULL) {
				prev->next = new_block;
			}

			if (next != NULL) {
				next->prev = new_block;
			}

			new_block->prev = prev;
			new_block->next = next;

			int copy_size = (row_num_ > new_max_row_num) ? new_max_row_num : row_num_;
			int block_x = (copy_size > BLOCK_SIZE_X) ? BLOCK_SIZE_X : copy_size;
			int grid_x = (copy_size - 1) / block_x + 1;
			int grid_y = col_num_;

			dim3 block(block_x, 1, 1);
			dim3 grid(grid_x, grid_y, 1);

			// Copy the upper half of the table to the new block
			copyTable<<<grid, block>>>(primary_storage_, new_block->primary_storage_, copy_size, col_num_, offset_, new_block->offset_, 0, 0);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			new_block->row_num_ = copy_size;
			new_block->col_num_ = col_num_ + 1;


			// Now deal with lower half
			copy_size = (row_num_ > new_max_row_num) ? row_num_ - new_max_row_num : 0;

			if (copy_size > 0) {
				GBlock *new_block2 = new GBlock;

				new_block2->max_col_num_ = new_max_col_num;
				new_block2->max_row_num_ = new_max_row_num;
				new_block2->offset_ = new_max_row_num;

				block_x = (copy_size > BLOCK_SIZE_X) ? BLOCK_SIZE_X : copy_size;
				grid_x = (copy_size - 1) / block_x + 1;
				grid_y = col_num_;

				block.x = block_x;
				grid.x = grid_x;
				grid.y = grid_y;

				copyTable<<<grid, block>>>(primary_storage_, new_block2->primary_storage_, copy_size, col_num_, offset_, new_block2->offset_, new_max_row_num, 0);
				checkCudaErrors(cudaGetLastError());
				checkCudaErrors(cudaDeviceSynchronize());

				new_block2->row_num_ = copy_size;
				new_block2->col_num_ = col_num_ + 1;

				new_block2->prev = new_block;
				new_block2->next = new_block->next;

				if (new_block->next != NULL) {
					new_block->next->prev = new_block2;
				}
				new_block->next = new_block2;
			}

			prev = next = NULL;

			checkCudaErrors(cudaFree(primary_storage_));
			primary_storage_ = NULL;
			if (secondary_storage_ != NULL) {
				checkCudaErrors(cudaFree(secondary_storage_));
				secondary_storage_ = NULL;
			}
		}
	} else {
		if (col_num_ == max_col_num_) {
			int new_max_row_num, new_max_col_num;

			new_max_col_num = max_col_num_ + ADDITIONAL_COLS;
			new_max_row_num = MAX_ROWS_PER_BLOCK / new_max_col_num;

			// One new block for the upper half
			GBlock *new_block = new GBlock;

			new_block->max_col_num_ = new_max_col_num;
			new_block->max_row_num_ = new_max_row_num;
			new_block->offset_ = new_max_row_num;

			// Replace the current block with the new block
			if (prev != NULL) {
				prev->next = new_block;
			}

			if (next != NULL) {
				next->prev = new_block;
			}

			new_block->prev = prev;
			new_block->next = next;

			int copy_size = (row_num_ > new_max_row_num) ? new_max_row_num : row_num_;

			// Copy the upper half of the table to the new block
			copyTable<<<grid, block>>>(primary_storage_, new_block->primary_storage_, copy_size, col_num_, offset_, new_block->offset_, 0, 0);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			new_block->row_num_ = copy_size;
			new_block->col_num_ = col_num_ + 1;


			// Now deal with lower half
			copy_size = (row_num_ > new_max_row_num) ? row_num_ - new_max_row_num : 0;

			if (copy_size > 0) {
				GBlock *new_block2 = new GBlock;

				new_block2->max_col_num_ = new_max_col_num;
				new_block2->max_row_num_ = new_max_row_num;
				new_block2->offset_ = new_max_row_num;

				block_x = (copy_size > BLOCK_SIZE_X) ? BLOCK_SIZE_X : copy_size;
				grid_x = (copy_size - 1) / block_x + 1;
				grid_y = col_num_;

				block.x = block_x;
				grid.x = grid_x;
				grid.y = grid_y;

				copyTable<<<grid, block>>>(primary_storage_, new_block2->primary_storage_, copy_size, col_num_, offset_, new_block2->offset_, new_max_row_num, 0);
				checkCudaErrors(cudaGetLastError());
				checkCudaErrors(cudaDeviceSynchronize());

				new_block2->row_num_ = copy_size;
				new_block2->col_num_ = col_num_ + 1;

				new_block2->prev = new_block;
				new_block2->next = new_block->next;

				if (new_block->next != NULL) {
					new_block->next->prev = new_block2;
				}
				new_block->next = new_block2;
			}

			prev = next = NULL;

			checkCudaErrors(cudaFree(primary_storage_));
			primary_storage_ = NULL;
			if (secondary_storage_ != NULL) {
				checkCudaErrors(cudaFree(secondary_storage_));
				secondary_storage_ = NULL;
			}
		}
	}

	return true;
}

bool GBlock::addRow()
{

}

GBlock::~GBlock()
{
	if (primary_storage_ != NULL) {
		checkCudaErrors(cudaFree(primary_storage_));
		primary_storage_ = NULL;
	}

	if (secondary_storage_ != NULL) {
		checkCudaErrors(cudaFree(secondary_storage_));
		secondary_storage_ = NULL;
	}

	schema_ = NULL;
}

}
