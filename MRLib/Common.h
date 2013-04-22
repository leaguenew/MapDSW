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
using namespace std;
/**
 * important specs
 */
struct Specs{

Specs(int argc, char** argv);
void printArg();

//for input data, offsets of the input data
	const void* offsets;
	unsigned int input_size ;
	unsigned int unit_size;

//global data
	const global_data_t* gbdata;






};

struct Job{
	Job();
	const void* input;
	unsigned int input_size;
};

struct Output{

};

#endif /* COMMON_H_ */
