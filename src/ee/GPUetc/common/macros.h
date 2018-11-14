#ifndef GPUTUPLE_H
#define GPUTUPLE_H

namespace gpu {

#define DEFAULT_PART_SIZE_ (1024 * 1024)
#define PART_SIZE_ 1024
#define BLOCK_SIZE_X_ 1024
#define MAX_BLOCK_SIZE_ (32 * 1024 * 1024)
#define MAX_STACK_SIZE_ 32


typedef struct _RESULT {
    int lkey;
    int rkey;
} RESULT;

typedef struct _RESULT_BOUND {
	int outer;
	int left;
	int right;
} ResBound;


}

#endif
