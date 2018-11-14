#include <cuda.h>
#include <cuda_runtime.h>
#include "KeyIndex.h"
#include "TreeIndex.h"

namespace gpu {

GTreeIndex::GTreeIndex() {
	key_schema_ = NULL;
	sorted_idx_ = NULL;
	key_idx_ = NULL;
	key_size_ = 0;
	key_num_ = 0;
	packed_key_ = NULL;

	checkCudaErrors(cudaMalloc(&sorted_idx_, sizeof(int) * DEFAULT_PART_SIZE_));	//Default 1024 * 1024 entries
}


GTreeIndex::GTreeIndex(int key_size, int key_num)
{
	key_size_ = key_size;
	key_num_ = key_num;

	checkCudaErrors(cudaMalloc(&sorted_idx_, sizeof(int) * key_num_));
	checkCudaErrors(cudaMalloc(&packed_key_, sizeof(int64_t) * key_num_ * key_size_));
	checkCudaErrors(cudaMalloc(&key_schema_, sizeof(GColumnInfo) * key_size_));
	checkCudaErrors(cudaMalloc(&key_idx_, sizeof(int) * key_size_));
}

GTreeIndex::GTreeIndex(int *sorted_idx, int *key_idx, int key_size, int64_t *packed_key, GColumnInfo *key_schema, int key_num)
{
	sorted_idx_ = sorted_idx;
	key_idx_ = key_idx;
	key_size_ = key_size;
	packed_key_ = packed_key;
	key_schema_ = key_schema;
	key_num_ = key_num;
}

__global__ void setKeySchema(GColumnInfo *key_schema, GColumnInfo *table_schema, int *key_idx, int key_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < key_size; i += stride) {
		key_schema[i] = table_schema[key_idx[i]];
	}
}

__global__ void initialize(GTreeIndex table_index, int64_t *table, GColumnInfo *schema, int columns, int left, int right)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index + left; i <= right; i += stride) {
		GTuple tuple(table + i * columns, schema, columns);

		table_index.insertKeyTupleNoSort(tuple, i);
	}
}

/* Sort chunks of elements on the same array. The min and max boundaries are specified by
 * left and right.
 * Chunk_size specifies size of each chunk and is doubled after each time of combination.
 */
__global__ void mergeSort(GTreeIndex table_index, int left, int right, int *input, int *output, int chunk_size)
{
	int half_size = (right - left)/2 + 1;

	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < half_size; i += blockDim.x * gridDim.x) {
		int lkey_id = left + i;
		int rkey_id = lkey_id + half_size;
		int lchunk_id = left + (i/chunk_size) * chunk_size;
		int rchunk_id = lchunk_id + half_size;
		int lsize = chunk_size;
		int rsize = (rchunk_id + chunk_size <= right) ? chunk_size : right - right_root + 1;
		int output_chunk_id = left + (i / (chunk_size * 2)) * chunk_size * 2;
		int llocal_idx = lkey_id - lchunk_id;
		int rlocal_idx = rkey_id - rchunk_id;

		// Extract key values
		GTreeIndexKey lkey = table_index.getKeyAtSortedIndex(lkey_id);
		GTreeIndexKey rkey = table_index.getKeyAtSortedIndex(rkey_id);
		// Find the lower bound of lkey on the right chunk
		int new_lkey_id = table_index.lowerBound(lkey, rchunk_id, rchunk_id + rsize - 1);
		// Find the upper bound of rkey on the left chunk
		int new_rkey_id = table_index.upperBound(rkey, lchunk_id, lchunk_id + lsize - 1);

		new_lkey_id = new_lkey_id - rchunk_id + llocal_idx;
		new_rkey_id = new_rkey_id - lchunk_id + rlocal_idx;

		output[new_left_key_ptr + base_idx] = input[left_key_ptr];
		output[new_right_key_ptr + base_idx] = input[right_key_ptr];
	}
}

__global__ void mergeSort2(GTreeIndex table_index, int left, int right, int *input, int *output, int chunk_size)
{
	int chunk_num = (right - left) / 2 + 1;
	int middle_chunk_id = (chunk_num / 2) * chunk_size + left;
	int half_size = middle_chunk_id - left;

	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < (right - left + 1); i += blockDim.x * gridDim.x) {
		int key_id = left + i;
		int local_chunk_id = key_id + (i / chunk_size) * chunk_size;
		int foreign_chunk_id = (local_chunk_id < middle_chunk_id) ? local_chunk_id + middle_chunk_id : local_chunk_id - middle_chunk_id;
		int size = (foreign_chunk_id + chunk_size <= right) ? chunk_size : right - foreign_chunk_id + 1;
		int output_chunk_id = (((local_chunk_id - left) % half_size) / (chunk_size * 2)) * chunk_size * 2;

		GTreeIndexKey key = table_index.getKeyAtSortedIndex(key_id);

		int new_key_id;

		if (key_id < middle_chunk_id) {
			new_key_id = table_index.lowerBound(key, foreign_chunk_id, foreign_chunk_id + size - 1);
		} else {
			new_key_id = table_index.upperBound(key, foreign_chunk_id, foreign_chunk_id + size - 1);
		}

		new_key_id = new_key_id - foreign_chunk_id + key_id - local_chunk_id + output_chunk_id;

		output[new_key_id] = input[lkey_id];
	}
}

void GTreeIndex::createIndex(int64_t *table, GColumnInfo *schema, int rows, int columns)
{
	key_num_ = rows;

	int block_x = (key_num_ < BLOCK_SIZE_X) ? key_num_ : BLOCK_SIZE_X;
	int grid_x = (key_num_ - 1) / block_x + 1;

	setKeySchema<<<grid_x, block_x>>>(key_schema_, schema, key_idx_, key_size_);

	GTreeIndex current_index(sorted_idx_, key_idx_, key_size_, packed_key_, key_schema_, key_num_);

	block_x = (key_num_ < BLOCK_SIZE_X) ? key_num_ : BLOCK_SIZE_X;
	grid_x = (key_num_ - 1)/block_x + 1;
	initialize<<<grid_x, block_x>>>(current_index, table, schema, columns, 0, key_num_ - 1);

	int *tmp_sorted_idx, *tmp;

	checkCudaErrors(cudaMalloc(&tmp_sorted_idx, sizeof(int) * key_num_));

	for (int chunk_size = 1; chunk_size <= key_num_/2; chunk_size <<= 1) {
		mergeSort2<<<grid_x, block_x>>>(*this, 0, key_num_ - 1, sorted_idx_, tmp_sorted_idx, chunk_size);
		checkCudaErrors(cudaDeviceSynchronize());

		tmp = sorted_idx_;
		sorted_idx_ = tmp_sorted_idx;
		tmp_sorted_idx = tmp;
	}

	checkCudaErrors(cudaFree(tmp_sorted_idx));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void upperBoundSearch(GTreeIndex indexes, GTuple new_tuple, int *key_schema, int key_size, int *entry_idx)
{
	int key_num = indexes.getKeyNum();
	GTreeIndexKey key = indexes.getKeyAtIndex(key_num);

	key.createKey(new_tuple, key_schema, key_size);

	*entry_idx = indexes.upperBound(key);
}

__global__ void lowerBoundSearch(GTreeIndex indexes, GTuple new_tuple, int *key_schema, int key_size, int *entry_idx)
{
	int key_num = indexes.getKeyNum();
	GTreeIndexKey key = indexes.getKeyAtIndex(key_num);

	key.createKey(new_tuple, key_schema, key_size);

	*entry_idx = indexes.lowerBound(key);
}

void GTreeIndex::addEntry(GTuple new_tuple) {
	int entry_idx, *entry_idx_dev;

	GTreeIndex current_index(sorted_idx_, key_idx_, key_size_, packed_key_, key_schema_, key_num_);

	checkCudaErrors(cudaMalloc(&entry_idx_dev, sizeof(int)));
	upperBoundSearch<<<1, 1>>>(current_index, new_tuple, key_idx_, key_size_, entry_idx_dev);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(&entry_idx, entry_idx_dev, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(sorted_idx_ + entry_idx + 1, sorted_idx_ + entry_idx, sizeof(int) * (key_num_ - entry_idx + 1), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(sorted_idx_ + entry_idx, &key_num_, sizeof(int), cudaMemcpyHostToDevice));
	key_num_++;
}

/* Add multiple new indexes.
 * New table are already stored in table_ at indexes started from base_idx.
 *
 * */
void GTreeIndex::addBatchEntry(int64_t *table, GColumnInfo *schema, int rows, int columns)
{
	GTreeIndex new_index(sorted_idx_ + key_num_, key_idx_, key_size_, packed_key_ + key_num_ * key_size_, key_schema_, rows);

	new_index.createIndex(table, schema, rows, columns);

	merge(0, key_num_ - 1, key_num_, key_num_ + rows - 1);
	key_num_ += rows;
}

__global__ void batchSearchUpper(GTreeIndex indexes, int key_left, int key_right, int left, int right, int *output) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTreeIndexKey key;

	for (int i = index; i <= key_right - key_left + 1; i += stride) {
		key = indexes.getKeyAtIndex(i + key_left);

		output[i] = indexes.upperBound(key, left, right);
	}
}

//Search for the lower bounds of an array of keys
__global__ void batchSearchLower(GTreeIndex indexes, int key_left, int key_right, int left, int right, int *output) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTreeIndexKey key;

	for (int i = index; i <= key_right - key_left + 1; i += stride) {
		key = indexes.getKeyAtIndex(i + key_left);

		output[i] = indexes.lowerBound(key, left, right);
	}
}

__global__ void constructWriteLocation(int *location, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < size; i += stride) {
		location[i] += i;
	}
}

__global__ void rearrange(int *input, int *output, int *location, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < size; i+= stride) {
		output[location[i]] = input[i];
	}
}

/* Merge new array to the old array
 * Both the new and old arrays are already sorted
 */
void GTreeIndex::merge(int old_left, int old_right, int new_left, int new_right) {
	int old_size, new_size;

	old_size = old_right - old_left + 1;
	new_size = new_right - new_left + 1;

	int block_x, grid_x;

	block_x = (new_size < BLOCK_SIZE_X) ? new_size : BLOCK_SIZE_X;
	grid_x = (new_size - 1) / block_x + 1;

	GTreeIndex current_index(sorted_idx_, key_idx_, key_size_, packed_key_, key_schema_, key_num_);

	int *write_location;

	checkCudaErrors(cudaMalloc(&write_location, (old_size + new_size) * sizeof(int)));
	batchSearchUpper<<<grid_x, block_x>>>(current_index, new_left, new_right, old_left, old_right, write_location + old_size);
	constructWriteLocation<<<grid_x, block_x>>>(write_location + old_size, new_size);

	block_x = (old_size < BLOCK_SIZE_X) ? old_size : BLOCK_SIZE_X;
	grid_x = (old_size - 1)/block_x + 1;

	batchSearchLower<<<grid_x, block_x>>>(current_index, old_left, old_right, new_left, new_right, write_location);
	constructWriteLocation<<<grid_x, block_x>>>(write_location, old_size);

	block_x = (old_size + new_size < BLOCK_SIZE_X) ? (old_size + new_size) : BLOCK_SIZE_X;
	grid_x = (old_size + new_size - 1)/block_x + 1;

	int *new_sorted_idx;

	checkCudaErrors(cudaMalloc(&new_sorted_idx, (old_size + new_size) * sizeof(int)));
	rearrange<<<grid_x, block_x>>>(sorted_idx_, new_sorted_idx, write_location, old_size + new_size);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(sorted_idx_));
	sorted_idx_ = new_sorted_idx;
}

void GTreeIndex::removeIndex() {
	if (sorted_idx_ != NULL)
		checkCudaErrors(cudaFree(sorted_idx_));
	if (key_idx_ != NULL)
		checkCudaErrors(cudaFree(key_idx_));
}
}
