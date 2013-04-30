/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * Common.h
 *
 *  Created on: 2013-4-8
 *      Author: Shiwei Dong
 */

#ifndef COMMON_H_
#define COMMON_H_

#include "../UserDef/GlobalDS.h"

//===========================
// Some important Definitions
//===========================
#define WARP 32
#define CACHEGROUP 8
#define MEM_BUCKETS 16384
#define CACHE_BUCKETS 800
#define MEM_POOL 4*1024*1024
#define SM_POOL 1024


//=============================
// important data structures
//=============================
struct Specs{

Specs(int argc, char** argv);
void printArg();

//for input data, offsets of the input data
	const unsigned int* input;
	//gl
	unsigned int input_size ;
//	unsigned int unit_size;

//global data
	const global_data_t* gbdata;
	unsigned int gbdata_size;

//for the scheduler
	unsigned int dim_grid;
	unsigned int dim_block;

};

struct Job{
	Job();
	const unsigned int* input;
	unsigned int input_size;
//	unsigned int unit_size;
	unsigned int data_size;
};

struct Output{

};

struct Intermediate{
	//key
	//value
	unsigned int keysize;
	unsigned int valuesize;
};

//specs used in GPU
struct GpuSpecs{
	const unsigned int* input;
//	unsigned int input_size;
//	unsigned int unit_size;
};

#endif /* COMMON_H_ */
