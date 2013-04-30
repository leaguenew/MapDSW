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
	unsigned int buckets_remain;

	char memoryPool[CACHE_BUCKETS];


private:
	__device__ void* Cache_Alloc();
	__device__ void flush();
	__device__ bool canInsert();
	__device__ void insertOrUpdate(Intermediate *);


};


#endif /* SMCACHE_H_ */
