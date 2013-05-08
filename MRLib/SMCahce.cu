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

#include "assert.h"
#include "Common.h"
#include "SMCache.h"
#include "MemAlloc.h"
#include "Intermediate.h"
#include "../UtilLib/hash.h"
#include "../UtilLib/GpuUtil.h"
#include "../UserDef/Mapreduce.h"

__shared__ unsigned int domerge;

__device__ void SMCache::init() {

	for (int i = 0; i < CACHE_POOL; i++) {
		memoryPool[i] = 0;
	}
	for (int j = 0; j < CACHE_BUCKETS; j++) {
		key_index[j] = 0;
		value_index[j] = 0;
		key_size[j] = 0;
		value_size[j] = 0;
		lock[j] = 0;
	}
	buckets_remain = CACHE_BUCKETS;
	offset = 0;
}

__device__ void SMCache::flush() {
	for (int i = 0; i < CACHE_POOL; i++) {
		memoryPool[i] = 0;
	}
	for (int j = 0; j < CACHE_BUCKETS; j++) {
		key_index[j] = 0;
		value_index[j] = 0;
		key_size[j] = 0;
		value_size[j] = 0;
		lock[j] = 0;
	}
	buckets_remain = CACHE_BUCKETS;
	offset = 0;
}

/*Allocate memory from the SMCache memory pool. If success, return the offset. else return -1*/
__device__ int SMCache::Cache_Alloc(unsigned int size) {

	size = align(size, 4) / 4;

	if (buckets_remain > 0 && (offset + size) < CACHE_POOL) {
		unsigned int result = atomicAdd((unsigned int*) &offset, size);
		//double check if the offset does not overflow
		if (offset < CACHE_POOL) {
			return (int) result;
		}
		return -1;
	}
	return -1;
}

__device__ void* SMCache::getaddress(unsigned short offset) {
	return memoryPool + offset;
}

//get intermediate from cache buckets which is used while merged into the Mem_Alloc
__device__ bool SMCache::getIntermediate(Intermediate * result,
		unsigned int bucket) {

	assert(bucket<CACHE_BUCKETS);

	if (bucket < CACHE_BUCKETS && key_index[bucket] != 0) {
		unsigned short keysize = key_size[bucket];
		unsigned short valuesize = value_size[bucket];

		result->init(getaddress(key_index[bucket]), keysize,
				getaddress(value_index[bucket]), valuesize);

		return true;
	}

	return false;
}

/**
 * This function perform as a insert and update function in SMCache.
 * The input is the intermediate date which is emitted at the end of the Map function
 */
__device__ void SMCache::insert(Intermediate *inter, MemAlloc* mem_alloc_d, bool isFail) {

	//fyi:there is a global flag "domerge" to judge whether to merge or not
	 isFail = insertOrUpdate(inter);

	if (isFail == true) {
		atomicCAS(&domerge, 0, 1);
	}



}

/**
 * insert or update the value, if success return true, else return false
 */
//key maybe not correct
__device__ bool SMCache::insertOrUpdate(Intermediate* inter) {

	//hash the key in order to store the intermediate key value
	unsigned int hash_result = hash((void*) inter->key, inter->keysize);
	unsigned int result_bucket = hash_result % CACHE_BUCKETS;

	bool rehash = false;

	while (buckets_remain > MAX_REMAIN_BUCKETS_C) {

		//if the key's hash bucket does not contain a value, allocate sm memory to it and store the key, value, keysize and value size
		if (key_index[result_bucket] == 0) {

			//attention: should get lock in order to prevent multiple access to the same bucket at the same time
			if (getLock(&lock[result_bucket])) {

				//alloc space for key,value, and store the key in the memory allocated
				//trick: put tmp_offset_value first so that tmp_offset_key cannot be 0

				int tmp_offset_value = Cache_Alloc(inter->valuesize);
				int tmp_offset_key = Cache_Alloc(inter->keysize);

				//if the alloc failed return false
				if (tmp_offset_key < 0 || tmp_offset_value < 0) {
					assert(releaseLock(&lock[result_bucket]));
					return false;
				}

				key_index[result_bucket] = (unsigned short) tmp_offset_key;
				void* key_adress = getaddress((unsigned short) tmp_offset_key);
				copyVal(key_adress, (void*) inter->key, inter->keysize);

				value_index[result_bucket] = (unsigned short) tmp_offset_value;
				void* value_adress = getaddress(
						(unsigned short) tmp_offset_value);
				copyVal(value_adress, (void*) inter->value, inter->valuesize);

				key_size[result_bucket] = inter->keysize;
				value_size[result_bucket] = inter->valuesize;
				atomicSub(&buckets_remain, 1);

				assert(releaseLock(&lock[result_bucket]));

				return true;
			}
			rehash = true;

		} else {

			//else when conflict

			unsigned short currentKeysize = key_size[result_bucket];
			unsigned short currentKeyindex = key_index[result_bucket];

			if (inter->keysize == currentKeysize) {
				char *currentkey = (char*) getaddress(currentKeyindex);
				if (compare(currentkey, inter->key, currentKeysize)) {
					//the current key is exactly the same as the input key, do the reduce step and update the value
					reduce();
					return true;
				} else {
					//the current key is not the same, then rehash
					rehash = true;
				}
			} else {
				rehash = true;
			}
		}
		if (rehash == true) {
			result_bucket = (result_bucket + 1) % CACHE_BUCKETS;
			rehash = false;
		}
	}

	return false;
}

