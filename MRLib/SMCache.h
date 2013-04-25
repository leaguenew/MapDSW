/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * SMCache.h
 *
 *  Created on: 2013-4-21
 *      Author: Shiwei Dong
 */

#ifndef SMCACHE_H_
#define SMCACHE_H_

/**
 * SMcache is used in GPU shared memory
 */

class SMCache{
public:
	//interface
	__device__ void init();
	__device__ void insert();


private:
	enum CacheMode=["a","b","c"];

private:
	__device__ void flush();
};


#endif /* SMCACHE_H_ */
