/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * GpuUtil.h
 *
 *  Created on: 2013-4-23
 *      Author: Shiwei Dong
 */
#ifndef GPUUTIL_H
#define GPUUTIL_H

#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <stdio.h>

#  define CUT_CHECK_ERROR(errorMessage) do {				 \
    cudaError_t err = cudaGetLastError();				    \
    if( cudaSuccess != err) {						\
	fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
		errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
	exit(EXIT_FAILURE);						  \
    }									\
    err = cudaThreadSynchronize();					   \
    if( cudaSuccess != err) {						\
	fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
		errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
	exit(EXIT_FAILURE);						  \
    } } while (0)

#  define CE(call) do {                                \
	call;CUT_CHECK_ERROR("------- Error ------\n"); \
     } while (0)


__device__ uint32_t getThreadID(){
//	uint32_t block_id=blockIdx.y*gridDim.x+blockIdx.x;
//	uint32_t blockSize=blockDim.z*blockDim.y*blockDim.x;
//	uint32_t thread_id=threadIdx.z*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x;
//	return block_id*blockSize+thread_id;
	return blockIdx.x*blockDim.x+threadIdx.x;
}

__device__ uint32_t getNumThreads(){
	return (gridDim.y*gridDim.x)*(blockDim.x*blockDim.y*blockDim.z);
}

__device__ unsigned int align(unsigned int size, unsigned int ALIGN)
{
	return (size+ALIGN-1)&(~(ALIGN-1));
}

__device__ void copyVal(void *dst, void *src, unsigned short size)
{
	 char *d=(char*)dst;
	 const char *s=(const char *)src;
	 for(unsigned short i=0;i < size;i++)
		 d[i]=s[i];
}

//the bucket is locked when lock==1, initially 0
__device__ bool getLock(int* lock) {
	return atomicCAS(lock, 0, 1) == 0;
}

//the bucket is locked when lock==1, initially 0
__device__ bool releaseLock(int* lock) {
	return atomicCAS(lock, 1, 0) == 1;

}

__host__ void memcpyD2H(void * dst, const void * src, unsigned int size){
	CE(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

__host__ void memcpyH2D(void * dst, const void * src, unsigned int size){
	CE(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}


#endif
