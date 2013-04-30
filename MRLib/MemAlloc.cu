/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * MemAlloc.cu
 *
 *  Created on: 2013-4-15
 *      Author: Shiwei Dong
 */

#include "MemAlloc.h"


void MemAlloc::init(){
	offset=0;
}

__device__ void MemAlloc::Start_MA_kernal(){
	//the first thread in each block get some memory space from the memory allocator
	unsigned int tid=threadIdx.x;
	unsigned int bid=blockIdx.x;
	if(tid==0){
		//the first memory address of each warp of a block
		//is stored in the shared memory to accelerate access
		global_mem_offset[]
	}
}

__device__ void Merge_SMCahce(SMCache* ){
	//This funtion should fully utilize every thread
	//every thread in a group help merge a number of SMCache buckets into the global memory
}
