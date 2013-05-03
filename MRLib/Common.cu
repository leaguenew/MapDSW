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




void Specs::printArg(){
	cout<<"The Grid Dimension is: "<<dim_grid<<endl;
	cout<<"The Block Dimension is: "<<dim_block<<endl;
}

//parse the parameters from the command line and store it into the Spec.
Specs::Specs(int argc, char** argv){
//for test
	dim_grid=128;
	dim_block=256;
}
