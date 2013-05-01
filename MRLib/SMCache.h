/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * SMCache.h
 *
 *  Created on: 2013-4-21
 *      Author: Shiwei Dong
 */

#ifndef SMCACHE_H_
#define SMCACHE_H_

#include "Common.h"
/**
 * SMcache is used in GPU shared memory
 */

class MemAlloc;

__shared__ unsigned int domerge;

class SMCache{
public:
	//interface
	__device__ void init();
	__device__ void insert(Intermediate *, MemAlloc* );


private:
	//different cache mode
	//enum CacheMode=["a","b","c"];
	//the key_index array store the index of each key in the memory pool
	unsigned short key_index[CACHE_BUCKETS];
	unsigned short value_index[CACHE_BUCKETS];
	unsigned short key_size[CACHE_BUCKETS];
	unsigned short value_size[CACHE_BUCKETS];
	unsigned int buckets_remain;

    unsigned int offset;
	char memoryPool[CACHE_POOL];

	int lock[CACHE_BUCKETS];

private:
	__device__ int Cache_Alloc(unsigned int size);
	__device__ void* getadress(unsigned int offset);
	__device__ void getvalue(void* address, unsigned int size)

	__device__ void flush();
	__device__ bool insertOrUpdate(Intermediate *);


};


#endif /* SMCACHE_H_ */
