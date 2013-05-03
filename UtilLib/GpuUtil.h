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

extern __device__ uint32_t getThreadID();
extern __device__ uint32_t getNumThreads();
extern __device__ unsigned int align(unsigned int size, unsigned int ALIGN);
extern __device__ void copyVal(void *dst, void *src, unsigned short size);
extern __device__ bool getLock(int* lock) ;
extern __device__ bool releaseLock(int* lock);
extern __host__ void memcpyD2H(void * dst, const void * src, unsigned int size);
extern __host__ void memcpyH2D(void * dst, const void * src, unsigned int size);


#endif
