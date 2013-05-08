/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * Mem_paras.h
 *
 *  Created on: 2013-5-8
 *      Author: Shiwei Dong
 */

#ifndef MEM_PARAS_H_
#define MEM_PARAS_H_

//===========================
// Some important Definitions
//===========================
#define DIM_GRID 16
#define DIM_BLOCK 256

#define WARP 32

#define MEM_BUCKETS 16384
#define MEM_POOL 4*1024*1024

//12*CACHEBUCKETS+4*CACHEPOOL+8 < 48kb/group
//the Cache_buckets should better bigger than Threads_in_block/CacheGroup.
//eg,bigger than 32 otherwise there may be unknown consequences
#define CACHEGROUP 8
#define CACHE_BUCKETS 250
#define CACHE_POOL 750


//the max remain buckets in SM Cache
//While the remain bucket is small, it will use more time to find an empty bucket which may waste time
#define MAX_REMAIN_BUCKETS_C 10  //eg. 20


#endif /* MEM_PARAS_H_ */
