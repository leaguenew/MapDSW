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

#include "cuda.h"

__constant__ char* memoryPool;

/**
 * part1: Reserved for
 */
class MemAlloc{
public:
	//Interface
	void init();

private:

	__device__ ;

private:



};


#endif /* MEMALLOC_H_ */
