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
#define _DEBUG


//=============================
// important data structures
//=============================
struct Specs {

	Specs(int argc, char** argv);
	void printArg();

//for input data, offsets of the input data
	const unsigned int* input;
//the number of input records* sizeof(int)
	unsigned int input_size;

//global data
	const global_data_t* gbdata;
	unsigned int gbdata_size;

//for the scheduler
	unsigned int dim_grid;
	unsigned int dim_block;

};

struct Job {
	const unsigned int* input;
	unsigned int input_size;
	unsigned int data_size;
};

struct Output {
	char *output_keys;
	char *output_vals;
	unsigned int *key_index;
	unsigned int *val_index;
	unsigned int count;
};

//for debug
#define bugbug(a) if(threadIdx.x==1){printf("%s:arrived here\n",a);}

//=====================================
// Print information
//=====================================
#ifdef _DEBUG
#define DoLog(...) do{printf(__VA_ARGS__);printf("\n");}while(0)
#else
#define DoLog(...)
#endif

#endif /* COMMON_H_ */
