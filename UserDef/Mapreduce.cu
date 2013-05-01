/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * Mapreduce.cu
 *
 *  Created on: 2013-4-3
 *      Author: Shiwei Dong
 */


/**
 * String match
 */
__device__ Intermediate map(void* Key, void* Value){

	//Match the key word

	return emit_intermediate();
}

__device__ void reduce(){


}

/**
 * compare the value, Most time, this function does not need to be changed
 */
__device__ bool compare(const void* a, const void* b, unsigned short size){
	return false;
}
