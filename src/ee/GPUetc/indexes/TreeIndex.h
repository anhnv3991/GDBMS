#ifndef TREE_INDEX_H_
#define TREE_INDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "KeyIndex.h"
#include "Index.h"
#include "GPUetc/storage/gtuple.h"
#include "GPUetc/common/common.h"

namespace gpu {

/* Class for index keys.
 * Each index key contains multiple column values.
 * The value type of each value is indicated in schema_.
 */

class GTreeIndexKey: public GKeyIndex {
public:
	CUDAD GTreeIndexKey() {
		schema_ = NULL;
		size_ = 0;
		packed_key_ = NULL;
		offset_ = 0;
	}

	/* Construct a key object from a packed key buffer an a schema buffer.
	 */
	CUDAD GTreeIndexKey(int64_t *packed_key, GColumnInfo *schema, int key_size, int offset) {
		packed_key_ = packed_key;
		schema_ = schema;
		size_ = key_size;
		offset_ = offset;
	}

	CUDAD void createKey(int64_t *tuple, GColumnInfo *schema, int *key_schema, int key_size) {
		for (int i = 0; i < key_size; i++) {
			packed_key_[i * offset_] = tuple[key_schema[i]];
			schema_[i * offset_] = schema[key_schema[i]];
		}
	}

	CUDAD void createKey(GTuple tuple, int *key_schema, int key_size) {
		for (int i = 0; i < key_size; i++) {
			packed_key_[i * offset_] = tuple.tuple_[key_schema[i]];
			schema_[i * offset_] = tuple.schema_[key_schema[i]];
		}
	}

	CUDAD void createKey(int64_t *tuple, GColumnInfo *schema) {
		for (int i = 0; i < size_; i++) {
			packed_key_[i * offset_] = tuple[i];
			schema_[i * offset_] = schema[i];
		}
	}

	CUDAD void createKey(GTuple tuple) {
		for (int i = 0; i < size_; i++) {
			packed_key_[i * offset_] = tuple.tuple_[i];
			schema_[i * offset_] = tuple.schema_[i];
		}
	}

	static CUDAD int KeyComparator(GTreeIndexKey left, GTreeIndexKey right) {
		int64_t res_i = 0;
		double res_d = 0;
		ValueType left_type, right_type;

		for (int i = 0; i < right.size_ && res_i == 0 && res_d == 0; i++) {
			left_type = left.schema_[i].data_type;
			right_type = right.schema_[i].data_type;

			if (left_type != VAL_INVALID && right_type != VAL_INVALID
					&& left_type != VAL_NULL && right_type != VAL_NULL) {
				int64_t left_i = (left_type == VAL_DOUBLE) ? 0 : left.packed_key_[i * left.offset_];
				int64_t right_i = (right_type == VAL_DOUBLE) ? 0 : right.packed_key_[i * right.offset_];
				double left_d = (left.schema_[i].data_type == VAL_DOUBLE) ? *reinterpret_cast<double *>(left_i) : static_cast<double>(left_i);
				double right_d = (right.schema_[i].data_type == VAL_DOUBLE) ? *reinterpret_cast<double *>(right_i) : static_cast<double>(right_i);

				res_i = (left_type == VAL_DOUBLE || right_type == VAL_DOUBLE) ? 0 : (left_i - right_i);
				res_d = (left_type == VAL_DOUBLE || right_type == VAL_DOUBLE) ? (left_d - right_d) : 0;
			}
		}

		return (res_i > 0 || res_d > 0) ? 1 : ((res_i < 0 || res_d < 0) ? -1 : 0);
	}

	/* Insert a key value to the key tuple at a specified key column.
	 * The value of the key value is of type int64_t.
	 * This is used to construct key values of a tuple.
	 * The type of the key value is ignored.
	 */
	CUDAD void insertKeyValue(int64_t value, int key_col) {
		packed_key_[key_col * offset_] = value;
	}

private:
	GColumnInfo *schema_;
	int64_t *packed_key_;
	int offset_;
};

/* Class for tree index.
 * Each index contains a list of key values and a sorted index array.
 * A schema array indicate the type of each key value.
 */
class GTreeIndex {
	friend class GTreeIndexKey;
public:
	GTreeIndex();
	GTreeIndex(int key_size, int key_num);
	GTreeIndex(int *sorted_idx, int *key_idx, int key_size, int64_t *packed_key, GColumnInfo *key_schema, int key_num);

	void createIndex(int64_t *table, GColumnInfo *schema, int rows, int columns);

	void addEntry(GTuple new_tuple);

	void addBatchEntry(int64_t *table, GColumnInfo *schema, int rows, int columns);

	void merge(int old_left, int old_right, int new_left, int new_right);


	CUDAD GTreeIndexKey getKeyAtSortedIndex(int key_index) {
		return GTreeIndexKey(packed_key_ + sorted_idx_[key_index] * key_size_, key_schema_, key_size_);
	}

	CUDAD GTreeIndexKey getKeyAtIndex(int key_index) {
		return GTreeIndexKey(packed_key_ + key_index * key_size_, key_schema_, key_size_);
	}

	CUDAD GColumnInfo *getSchema();

	CUDAD int getKeyNum();

	CUDAD int *getSortedIdx();

	CUDAD int *getKeyIdx();

	CUDAD int getKeySize();

	CUDAD int64_t *getPackedKey();

	/* Largest element that is less than key and has the largest location id */
	CUDAD int largestStrictLesser(GTreeIndexKey key, int left, int right);
	/* Largest element that is less than or equal to key and has the largest location id */
	CUDAD int largestWeakLesser(GTreeIndexKey key, int left, int right);
	/* Smallest element that is larger than key and has the smallest location id */
	CUDAD int smallestStrictGreater(GTreeIndexKey key, int left, int right);
	/* Smallest element that is larger than or equal to key and has the smallest location id */
	CUDAD int smallestWeakGreater(GTreeIndexKey key, int left, int right);

	CUDAD int lowerBound(GTreeIndexKey key, int left, int right);
	CUDAD int lowerBound(GTreeIndexKey key);

	CUDAD int upperBound(GTreeIndexKey key, int left, int right);
	CUDAD int upperBound(GTreeIndexKey key);

	/* Insert key values of a tuple to the 'location' of the key list 'packed_key_'.
	 */
	CUDAD void insertKeyTupleNoSort(GTuple tuple, int location);
	CUDAD void swap(int left, int right);


	void removeIndex();
protected:
	int key_num_;		// Number of key values (equal to the number of rows), also the offset
	int *sorted_idx_;	// Sorted key indexes
	int *key_idx_;		// Index of columns selected as keys
	int key_size_;		// Number of columns selected as keys
	int64_t *packed_key_;
	GColumnInfo *key_schema_;	// Schemas of columns selected as keys
};

CUDAD GColumnInfo *GTreeIndex::getSchema() {
	return key_schema_;
}

CUDAD int GTreeIndex::getKeyNum() {
	return key_num_;
}

CUDAD int *GTreeIndex::getSortedIdx() {
	return sorted_idx_;
}

CUDAD int *GTreeIndex::getKeyIdx() {
	return key_idx_;
}

CUDAD int GTreeIndex::getKeySize() {
	return key_size_;
}

CUDAD int64_t *GTreeIndex::getPackedKey() {
	return packed_key_;
}

/* Largest element that is less than key and has the largest location id */
CUDAD int GTreeIndex::largestStrictLesser(GTreeIndexKey key, int left, int right)
{
	int middle = -1;
	int result = -1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		// If key <= middle_key, move left, otherwise move right
		right = (compare_res <= 0) ? (middle - 1) : right;
		left = (compare_res <= 0) ? left : (middle + 1);

		//
		result = (compare_res > 0) ? middle : result;
	}

	return result;
}

/* Largest element that is less than or equal to key and has the largest location id */
CUDAD int GTreeIndex::largestWeakLesser(GTreeIndexKey key, int left, int right)
{
	int middle = -1;
	int result = -1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res < 0) ? (middle - 1) : right;
		left = (compare_res < 0) ? left : (middle + 1);

		result = (compare_res >= 0) ? middle : result;
	}

	return result;
}

/* Smallest element that is larger than key and has the smallest location id */
CUDAD int GTreeIndex::smallestStrictGreater(GTreeIndexKey key, int left, int right)
{
	int middle = -1;
	int result = right - 1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res < 0) ? (middle - 1) : right;
		left = (compare_res < 0) ? left : (middle + 1);

		result = (compare_res < 0) ? middle : result;
	}

	return result;
}
/* Smallest element that is larger than or equal to key and has the smallest location id */
CUDAD int GTreeIndex::smallestWeakGreater(GTreeIndexKey key, int left, int right)
{
	int middle = -1;
	int result = right - 1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res <= 0) ? (middle - 1) : right;
		left = (compare_res <= 0) ? left : (middle + 1);

		result = (compare_res <= 0) ? middle : result;
	}

	return result;
}


/* Smallest weak greater */
CUDAD int GTreeIndex::lowerBound(GTreeIndexKey key, int left, int right)
{
	int middle = -1;
	int result = -1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res <= 0) ? (middle - 1) : right;
		left = (compare_res <= 0) ? left : (middle + 1);
		result = (compare_res <= 0) ? middle : result;
	}

	return result;
}

CUDAD int GTreeIndex::lowerBound(GTreeIndexKey key)
{
	int left = 0, right = key_num_ - 1;
	int middle = -1;
	int result = -1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res <= 0) ? (middle - 1) : right;
		left = (compare_res <= 0) ? left : (middle + 1);
		result = (compare_res <= 0) ? middle : result;
	}

	return result;
}

/* Smallest strict greater */
CUDAD int GTreeIndex::upperBound(GTreeIndexKey key, int left, int right)
{
	int middle = -1;
	int result = right + 1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res < 0) ? (middle - 1) : right;
		left = (compare_res < 0) ? left : (middle + 1);
		result = (compare_res < 0) ? middle : result;
	}

	return result;
}

CUDAD int GTreeIndex::upperBound(GTreeIndexKey key)
{
	int left = 0, right = key_num_ - 1;
	int middle = -1;
	int result = right + 1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res < 0) ? (middle - 1) : right;
		left = (compare_res < 0) ? left : (middle + 1);
		result = (compare_res < 0) ? middle : result;
	}

	return result;
}

CUDAD void GTreeIndex::insertKeyTupleNoSort(GTuple tuple, int location)
{
	for (int i = 0; i < key_size_; i++) {
		packed_key_[location * key_size_ + i] = tuple.tuple_[key_idx_[i]];
	}
	sorted_idx_[location] = location;
}

CUDAD void GTreeIndex::swap(int left, int right)
{
	int tmp = sorted_idx_[left];

	sorted_idx_[left] = sorted_idx_[right];
	sorted_idx_[right] = tmp;
}

}

#endif
