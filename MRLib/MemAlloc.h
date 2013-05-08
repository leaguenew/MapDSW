/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * MemAlloc.h
 *
 *  Created on: 2013-4-15
 *      Author: Shiwei Dong
 */

#ifndef MEMALLOC_H_
#define MEMALLOC_H_

#include "Mem_paras.h"
#include "Common.h"
class SMCache;
class Intermediate;



class MemAlloc{
public:

	//Interface
	__device__ void Start_MA_kernal();
	__device__ void Merge_SMCache(SMCache* , unsigned int groupid);
	__device__ void insert(Intermediate * );
    __device__ void* getaddress(unsigned int offset);

public:

	unsigned int key_index[MEM_BUCKETS];
	unsigned int value_index[MEM_BUCKETS];
	unsigned int key_size[MEM_BUCKETS];
	unsigned int value_size[MEM_BUCKETS];
	//if 1 then the bucket is used
	unsigned int used[MEM_BUCKETS];
	int lock[MEM_BUCKETS];

	unsigned int memoryPool[MEM_POOL];


private:

    __device__ unsigned int Mem_Alloc(unsigned int size);
	__device__ bool insertOrUpdate(Intermediate *);


};


#endif /* MEMALLOC_H_ */
