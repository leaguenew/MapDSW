///* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
// * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
// *
// * SMCahce.cu
// *
// *  Created on: 2013-4-23
// *      Author: Shiwei Dong
// */
//
//
//#include "SMCache.h"
//
////insert or update the value in SMCache
//__device__ void SMCache::insert(){
//	//if not overflow
//	//insert into the SMCache
//
//	//if SMCache overflows swap all the gid
//	//merge the cache to the global memory
//	mem_alloc_d.Merge-Cache(cache[gid]);
//	//flush the cache
//	flush();
//	cudasync();
//	//insert into the SMCache;
//
//}
//
//

#include "SMCache.h"

__device__ void SMCache::init() {
	buckets_remain = CACHE_BUCKETS;
}

/**
 * This function perform as a insert and update function in SMCache.
 * The input is the intermediate date which is emitted at the end of the Map function
 */
__device__ void SMCache::insert(Intermediate *inter, MemAlloc* mem_alloc_d) {

		/**
		 * if the SMCache is not full, operate the insertion or update
		 */
		//there should be a global flag to judge whether to merge or not
		bool flag = canInsert();
		if (flag == true) {
			insertOrUpdate(inter);
		}else{
			atomicCAS(&domerge,0,1);
		}
		__syncthreads();
		/**
		 * else if the SMCache is full, stop all the threads and then swap the SMCache out and merge to the
		 * memory allocator. Then flush the SMCache, and insert again
		 */
		if (domerge==true) {
			mem_alloc_d->Merge_SMCache(this);
			if (tid % gid == 0) {
				flush();
			}
			if(tid==0){
				atomicExch(&domerge,0);
			}
		}
		__syncthreads();

		//must assert the interkey and value larger than the Cache Pool
		insertOrUpdate(inter);
}

__device__ bool SMCache::canInsert() {
	//if the remaining buckets and the remaining memory pool can store the key and value and their size, then it's true

}

__device__ void SMCache::insertOrUpdate(Intermediate* inter) {
	//hash the key and get the prefix sum
	//if the key is not exist, allocate sm memory to it, and store the key, value, keysize and value size
	//if the key is already exist, then use user defined reduce function to reduce the key

}
