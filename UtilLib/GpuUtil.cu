/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * GpuUtil.cu
 *
 *  Created on: 2013-5-3
 *      Author: Shiwei Dong
 */

#include "GpuUtil.h"
#include "../MRLib/Common.h"

__device__ uint32_t getThreadID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ uint32_t getNumThreads() {
	return (gridDim.y * gridDim.x) * (blockDim.x * blockDim.y * blockDim.z);
}

__device__ unsigned int align(unsigned int size, unsigned int ALIGN) {

	return (size + ALIGN - 1) & (~(ALIGN - 1));
}

__device__ void copyVal(void *dst, void *src, unsigned short size) {

	char *d = (char*) dst;
	const char *s = (const char *) src;
	for (unsigned short i = 0; i < size; i++)
		d[i] = s[i];
}

//the bucket is locked when lock==1, initially 0
__device__ bool getLock(int* lock) {
	return atomicCAS(lock, 0, 1) == 0;
}

//the bucket is locked when lock==1, initially 0
__device__ bool releaseLock(int* lock) {
	return atomicCAS(lock, 1, 0) == 1;

}

__host__ void memcpyD2H(void * dst, const void * src, unsigned int size) {
	CE(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

__host__ void memcpyH2D(void * dst, const void * src, unsigned int size) {
	CE(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

