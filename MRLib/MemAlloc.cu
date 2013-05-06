/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * MemAlloc.cu
 *
 *  Created on: 2013-4-15
 *      Author: Shiwei Dong
 */

#include "assert.h"
#include "Common.h"
#include "MemAlloc.h"
#include "SMCache.h"
#include "Intermediate.h"
#include "../UtilLib/hash.h"
#include "../UtilLib/GpuUtil.h"
#include "../UserDef/Mapreduce.h"

//#include "sm_11_atomic_functions.h"

////the global data in the device memory
//global_data_t* global_data_d;
//
////the offset
//unsigned int* input_offset_d;
//unsigned int* input_size_d;

//every block has a copy of this shared array. Since global atomic access use too much time, use 8 copies of offsets
//each copy stores the start address for its warp
__shared__ volatile unsigned int global_mem_offset[8];

/**
 * This function serves for initialize purpose. It is invoked when a kernel is launched.
 */
__device__ void MemAlloc::Start_MA_kernal() {

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int blocknum = gridDim.x;
	unsigned int gid = tid / WARP;

	if (tid % WARP == 0) {
		global_mem_offset[gid] = MEM_POOL * (8 * bid + gid) / (blocknum * 8);
		//printf("bid %d gid %d offset %d: \n",bid,gid,global_mem_offset[gid]);
		//if(bid==0&&tid==64) printf("global_mem_offset %d \n", global_mem_offset[gid]);
	}
}

/*Allocate memory from the Memory Allocator memory pool. If success, return the offset. else return -1*/
__device__ unsigned int MemAlloc::Mem_Alloc(unsigned int size) {

	unsigned int tid = threadIdx.x;
	unsigned int gid = tid / WARP;
	unsigned int tmp_size=align(size,sizeof(unsigned int))/sizeof(unsigned int);

	//Attention: this place may cause overflow, thus Please use smaller jobs
	unsigned int result = atomicAdd((unsigned int *) &global_mem_offset[gid],
			tmp_size);

	return result;
}

__device__ void* MemAlloc::getaddress(unsigned int offset) {
	return memoryPool + offset;
}

__device__ void MemAlloc::Merge_SMCache(SMCache* Cache, unsigned int groupid) {
//This function should fully utilize every thread
//every thread in a group help merge a number of SMCache buckets into the global memory
//the threads in one cache group deal with one Cache merge

	unsigned int tid = threadIdx.x;
	unsigned int num_threads = blockDim.x;
	unsigned int threads_per_group = align(num_threads, CACHEGROUP) / CACHEGROUP;
	unsigned int gid = tid / threads_per_group;

	if (groupid == gid) {
		for (int i = tid % threads_per_group; i < CACHE_BUCKETS; i +=
				threads_per_group) {
			//printf("threads_per_group %d\n",threads_per_group);
			Intermediate result;
			//	if(tid==160)printf("Group :%d, i %d\n", gid, i);
			//if(gid==7){printf(" i %d tid %d\n",i, tid);}
			if (Cache->getIntermediate(&result, i)) {
				//printf("merging from SM to Device memory %d\n", threadIdx.x);
				//printf("ssssssssgid %d, i %d tid %d\n", gid, i, tid);
			//	if(blockIdx.x==0)printf("inter value result %d\n",(int)*((char*)result.value));
				insert(&result);
			}
		}
	}
}

//It is different from the SMCache insert
__device__ void MemAlloc::insert(Intermediate* inter) {
//It should be asserted that the memory allocator is able to hold all the emitted intermediate and results
//The volume of a job should be determined during the slicing procedure not here
	assert(insertOrUpdate(inter));
}

__device__ bool MemAlloc::insertOrUpdate(Intermediate* inter) {

//hash the key in order to store the intermediate key value
	unsigned int hash_result = hash((void*) inter->key, inter->keysize);
	unsigned int result_bucket = hash_result % MEM_BUCKETS;

	bool rehash = false;
	int count = 0;

//may overflow when all the buckets are full, please avoid such situation
//if can not find a bucket after 1000 rehash, then assumed that the buckets are full
	while (count < 1000) {

		//if the key's hash bucket does not contain a value, allocate sm memory to it and store the key, value, keysize and value size
		if (key_index[result_bucket] == 0) {

			//attention: should get lock in order to prevent multiple access to the same bucket at the same time
			if (getLock(&lock[result_bucket])) {

				//alloc space for key,value, and store the key in the memory allocated
				unsigned int tmp_offset_key = Mem_Alloc(
						(unsigned int) inter->keysize);
				unsigned int tmp_offset_value = Mem_Alloc(
						(unsigned int) inter->valuesize);
			//	if(blockIdx.x==3)printf("tmp_offset_key %d tmp_offset_value %d\n",tmp_offset_key,tmp_offset_value );
				//the allocations of key value offset are assumed to be successful, if overflow, there will be unknown runtime errors
				key_index[result_bucket] = tmp_offset_key;
				void* key_adress = getaddress(tmp_offset_key);
				copyVal(key_adress, (void*) inter->key,
						(unsigned short) inter->keysize);

				value_index[result_bucket] = tmp_offset_value;
				void* value_adress = getaddress(tmp_offset_value);
				copyVal(value_adress, (void*) inter->value,
						(unsigned short) inter->valuesize);
//				printf("inter->value %d value %d\n",*(unsigned int*)inter->value,*(unsigned int*)(char*)getaddress(value_index[result_bucket]) );

				key_size[result_bucket] = (unsigned int) inter->keysize;
				value_size[result_bucket] = (unsigned int) inter->valuesize;
				used[result_bucket] = 1;

				assert(releaseLock(&lock[result_bucket]));
				return true;
			}
			rehash = true;

		} else {
			//else when conflict
			//printf("conflict!!!!\n");
			//get the key from bucket, aware that every key or value is ended by \0 so that we can get the key or value easily		unsigned short currentKeysize = key_size[result_bucket];
//			unsigned int currentKeysize = key_size[result_bucket];
//			unsigned int currentKeyindex = key_index[result_bucket];
//
//			if (inter->keysize == currentKeysize) {
//				char *currentkey = (char*) getaddress(currentKeyindex);
//				if (compare(currentkey, inter->key, currentKeysize)) {
//					//the current key is exactly the same as the input key, do the reduce step and update the value
//					printf("same!!!!!!!\n");
//					reduce();
//					return true;
//				} else {
//					//the current key is not the same, then rehash
//					rehash = true;
//				}
//			} else {
//				rehash = true;
//			}
			rehash=true;

		}

		if (rehash == true) {
			result_bucket = (result_bucket + 1) % MEM_BUCKETS;
			rehash = false;
		}
		count++;
	}
	//printf("count is %d\n", count);
	return false;
}

