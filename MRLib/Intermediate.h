/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * Intermediate.h
 *
 *  Created on: 2013-5-2
 *      Author: Shiwei Dong
 */

#ifndef INTERMEDIATE_H_
#define INTERMEDIATE_H_


class Intermediate {
public:
	__device__ void init(void*, unsigned short, void*, unsigned short valuesize_in);

public:
	const void* key;
	const void* value;
	unsigned short keysize;
	unsigned short valuesize;
};


#endif /* INTERMEDIATE_H_ */
