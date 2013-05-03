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
//struct global_data_t;

extern __device__ void emit_intermediate(Intermediate* inter, SMCache* Cache,
		MemAlloc* Mem_Alloc);

extern __device__ void map(void* global_data_d, unsigned int offset,
		SMCache* Cache, MemAlloc* Mem_Alloc);

extern __device__ void reduce();

extern __device__ bool compare(const void* a, const void* b,
		unsigned short size);

#endif
