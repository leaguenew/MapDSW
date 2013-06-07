/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential

 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * Mapreduce.cu
 *
 *  Created on: 2013-4-3
 *      Author: Shiwei Dong
 */
#ifndef MAPREDUCE_H_
#define MAPREDUCE_H_


class Intermediate;
class SMCache;
class MemAlloc;

extern __device__ void emit_intermediate(Intermediate* inter,
		MemAlloc* Mem_Alloc, SMCache* Cache = NULL, bool* isFail = 0);

extern __device__ void map(void* global_data_d, unsigned int offset,
		MemAlloc* Mem_Alloc, SMCache* Cache=NULL, bool* isFail = 0);

extern __device__ void reduce(Intermediate* current, Intermediate* inter);

extern __device__ int compare(const void* a, const void* b,
		unsigned short size_a, unsigned short size_b);

#endif
