/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * Intermediate.cu
 *
 *  Created on: 2013-5-3
 *      Author: Shiwei Dong
 */



#include "Intermediate.h"
#include "stdio.h"

__device__ void Intermediate::init(void* key_in, unsigned short keysize_in, void* value_in,
		unsigned short valuesize_in){
	//printf("%c %d\n", *(char*)key_in, (int)valuesize_in);
	key = (const void*) key_in;
	value = (const void*) value_in;
	keysize = keysize_in;
	valuesize = valuesize_in;
};
