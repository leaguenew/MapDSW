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
#include "UtilLib/hash.cu"
#include "assert.h"

__device__ void SMCache::init() {
	for (int i = 0; i < CACHE_POOL; i++) {
		memoryPool[i] = 0;
	}
	for (int j=0; j < CACHE_BUCKETS;j++){
		key_index[j]=0;
		value_index[j]=0;
		key_size[j]=0;
		value_size[j]=0;
		lock[j]=0;
	}
	buckets_remain = CACHE_BUCKETS;
	offset = 0;
}

__device__ void SMCache::flush() {
	for (int i = 0; i < CACHE_POOL; i++) {
		memoryPool[i] = 0;
	}
	for (int j=0; j < CACHE_BUCKETS;j++){
		key_index[j]=0;
		value_index[j]=0;
		key_size[j]=0;
		value_size[j]=0;
		lock[j]=0;
	}
	buckets_remain = CACHE_BUCKETS;
	offset = 0;
}

/*Allocate memory from the SMCache memory pool. If success, return the offset. else return -1*/
__device__ int SMCache::Cache_Alloc(unsigned int size) {
	if (buckets_remain > 0 && (offset + size) < CACHE_POOL) {
		unsigned int result = atomicAdd(&offset, size);
		//double check if the offset does not overflow
		if (offset < CACHE_POOL) {
			return result;
		}
		return -1;
	}
	return -1;
}

__device__ void* SMCache::getadress(unsigned int offset) {
	return memoryPool + offset;
}

/**
 * This function perform as a insert and update function in SMCache.
 * The input is the intermediate date which is emitted at the end of the Map function
 */
__device__ void SMCache::insert(Intermediate *inter, MemAlloc* mem_alloc_d) {

	/**
	 * if the SMCache is not full, operate the insertion or update
	 */
	//there is a global flag "domerge" to judge whether to merge or not
	bool flag = insertOrUpdate(inter);
	if (flag == false) {
		atomicCAS(&domerge, 0, 1);
	}
	__syncthreads();

	/**
	 * else if the SMCache is full, stop all the threads and then swap the SMCache out and merge to the
	 * memory allocator. Then flush the SMCache, and insert again
	 */
	if (domerge) {
		mem_alloc_d->Merge_SMCache(this);
		if (tid % gid == 0) {
			flush();
		}
		if (tid == 0) {
			atomicExch(&domerge, 0);
		}
	}
	__syncthreads();

	//must assert the intermediate key and value larger than the Cache Pool
	if (flag == false) {
		assert(insertOrUpdate(inter));
	}
}

/**
 * insert or update the value, if success return true, else return false
 */
__device__ bool SMCache::insertOrUpdate(Intermediate* inter) {

	//hash the key in order to store the intermediate key value
	unsigned int hash_result = hash((void*) inter->key, inter->keysize);
	unsigned int result_bucket = hash_result % CACHE_BUCKETS;

	bool rehash = false;

	while (buckets_remain>MAX_REMAIN_BUCKETS_C) {

		//if the key's hash bucket does not contain a value, allocate sm memory to it and store the key, value, keysize and value size
		if (key_index[result_bucket] == 0) {

			//attention: should get lock in order to prevent multiple access to the same bucket at the same time
			if (getLock(&lock[result_bucket])) {

				//alloc space for key,value, and store the key in the memory allocated
				unsigned short tmp_offset_key = Cache_Alloc(inter->keysize);
				unsigned short tmp_offset_value = Cache_Alloc(inter->valuesize);

				//if the alloc failed return false
				if (tmp_offset_key < 0 || tmp_offset_value < 0) {
					return false;
				}

				key_index[result_bucket] = tmp_offset_key;
				void* key_adress = getadress(tmp_offset_key);
				copyVal(key_adress, (void*) inter->key, inter->keysize);

				value_index[result_bucket] = tmp_offset_value;
				void* value_adress = getadress(tmp_offset_value);
				copyVal(value_adress, (void*) inter->value, inter->valuesize);

				key_size[result_bucket] = inter->keysize;
				value_size[result_bucket] = inter->valuesize;

				assert(releaseLock(&lock[result_bucket]));
				return true;
			}
			rehash = true;

		} else {
			//else when conflict

			//get the key from bucket, aware that every key or value is ended by \0 so that we can get the key or value easily
			unsigned short currentKeysize = key_size[result_bucket];
			if (inter->keysize == currentKeysize) {
				char *currentkey = getadress(currentKeysize);
				if (compare(currentkey, inter->key, currentKeysize)) {
					//the current key is exactly the same as the input key, do the reduce step and update the value

				} else {
					//the current key is not the same, then rehash
					rehash = true;
				}
			} else {
				rehash = true;
			}

		}
		if (rehash == true) {
			result_bucket = (hash_result + 1) % CACHE_BUCKETS;
			rehash = false;
		}
	}

	return false;
}


