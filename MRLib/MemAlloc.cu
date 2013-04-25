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
	if(tid==0){

	}
}
