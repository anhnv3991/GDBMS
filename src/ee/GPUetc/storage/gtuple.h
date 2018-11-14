#ifndef GTUPLE_H_
#define GTUPLE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "gnvalue.h"
#include <string>

namespace gpu {

class GTuple {
	friend class GKeyIndex;
	friend class GTreeIndexKey;
	friend class GHashIndexKey;
	friend class GTreeIndex;
	friend class GExpression;
	friend class GTable;
public:
	CUDAH GTuple();
	CUDAH GTuple(int64_t *tuple, GColumnInfo *schema_buff, int max_columns, int offset, char *secondary_storage = NULL);

	CUDAH int getColumnCount();
	CUDAH GNValue getGNValue(int column_idx);
	CUDAH bool setGNValue(GNValue value, int column_idx);


	CUDAH GNValue at(int column_idx);
	CUDAH GNValue operator[](int column_idx);

	std::string debug() const;
protected:
	int64_t *tuple_;
	char *secondary_storage_;
	GColumnInfo *schema_;
	int columns_;
	int offset_;
};

CUDAH GTuple::GTuple()
{
	tuple_ = NULL;
	secondary_storage_ = NULL;
	schema_ = NULL;
	columns_ = 0;
	offset_ = 0;
}


CUDAH GTuple::GTuple(int64_t *tuple, GColumnInfo *schema_buff, int max_columns, int offset, char *secondary_storage)
{
	tuple_ = tuple;
	secondary_storage_ = secondary_storage;
	schema_ = schema_buff;
	columns_ = max_columns;
	offset_ = offset;
}

CUDAH int GTuple::getColumnCount()
{
	return columns_;
}

CUDAH bool GTuple::setGNValue(GNValue value, int column_idx)
{
	if (column_idx >= columns_)
		return false;

	tuple_[column_idx * offset_] = value.m_data_;

	return true;
}

CUDAH GNValue GTuple::getGNValue(int column_idx)
{
	if (column_idx < columns_)
		return GNValue(schema_[column_idx].data_type, tuple_[column_idx * offset_], secondary_storage_);

	return GNValue::getInvalid();
}

CUDAH GNValue GTuple::at(int column_idx)
{
	if (column_idx < columns_)
		return GNValue(schema_[column_idx].data_type, tuple_[column_idx * offset_], secondary_storage_);

	return GNValue::getInvalid();
}

CUDAH GNValue GTuple::operator[](int column_idx)
{
	if (column_idx < columns_)
		return GNValue(schema_[column_idx].data_type, tuple_[column_idx * offset_], secondary_storage_);

	return GNValue::getInvalid();
}
}

#endif
