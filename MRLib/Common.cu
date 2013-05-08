/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * Common.cpp
 *
 *  Created on: 2013-4-17
 *      Author: Shiwei Dong
 */

#include <iostream>
using namespace std;

#include "Common.h"
#include "Mem_paras.h"
#include "../UtilLib/CommonUtil.h"

void Specs::printArg() {
	cout << "The Grid Dimension is: " << DIM_GRID << endl;
	cout << "The Block Dimension is: " << DIM_BLOCK << endl;
}

//parse the parameters from the command line and store it into the Spec.
Specs::Specs(int argc, char** argv) {
//for test
//	if (!get_opt(argc, argv, "f", filename)) {
//		cout << "usage: " << argv[0] << " -f filename" << endl;
//		return 1;
//	}
	input=NULL;
	input_size=0;
	gbdata_size=0;

//	dim_grid = 1;
//	get_opt(argc,argv,"blocks",dim_grid);
//
//	dim_block = 256;
//	get_opt(argc,argv,"threads",dim_block);


}
