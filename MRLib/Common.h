/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * Common.h
 *
 *  Created on: 2013-4-8
 *      Author: Shiwei Dong
 */

#ifndef COMMON_H_
#define COMMON_H_

#include "../UserDef/GlobalDS.h"

//===========================
// Some important Definitions
//===========================
#define WARP 32
#define CACHEGROUP 1
#define MEM_BUCKETS 1638400
// the Cache_buckets should better bigger than Threads_in_block/CacheGroup.
//eg,bigger than 32 otherwise there may be unknown consequences
#define CACHE_BUCKETS 1600


#define MEM_POOL 16*1024*1024
#define CACHE_POOL 2000

//the max remain buckets in SM Cache or MemAlloc.
//While the remain bucket is small, it will use more time to find an empty bucket which may waste time
#define MAX_REMAIN_BUCKETS_C 0  //eg. 20
#define MAX_REMAIN_BUCKETS_M 0  //eg. MEM_BUCKETS/10


//=============================
// important data structures
//=============================
struct Specs {

	Specs(int argc, char** argv);
	void printArg();

//for input data, offsets of the input data
	const unsigned int* input;
//the number of input records* sizeof(int)
	unsigned int input_size;
//	unsigned int unit_size;

//global data
	const global_data_t* gbdata;
	unsigned int gbdata_size;

//for the scheduler
	unsigned int dim_grid;
	unsigned int dim_block;

};

struct Job {
	const unsigned int* input;
	unsigned int input_size;
//	unsigned int unit_size;
	unsigned int data_size;
};

struct Output {
	char *output_keys;
	char *output_vals;
	unsigned int *key_index;
	unsigned int *val_index;
	unsigned int count;
};

//specs used in GPU
struct GpuSpecs {
	const unsigned int* input;
//	unsigned int input_size;
//	unsigned int unit_size;
};

//for debug
#define bugbug(a) if(threadIdx.x==1){printf("%s:arrived here\n",a);}

#endif /* COMMON_H_ */
