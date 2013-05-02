/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * hash.cu
 *
 *  Created on: 2013-5-3
 *      Author: Shiwei Dong
 */

#include "hash.h"


__device__ unsigned short hash(void *key, unsigned short size)
{
	unsigned short hs = 5381;
	char *str = (char *)key;

	for(int i = 0; i<size; i++)
	{
		hs = ((hs << 5) + hs) + ((int)str[i]); /* hash * 33 + c */
	}
	return hs;
}
