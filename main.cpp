/*
 * main.cpp
 *
 *  Created on: 2013-4-3
 *      Author: Shiwei Dong
 */
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include "time.h"
#include "math.h"
#include <vector>
#include <assert.h>
#include <stdio.h>
using namespace std;

#include "MRLib/TaskScheduler.h"
#include "MRLib/Common.h"
#include "UtilLib/CommonUtil.h"

/**
 *
 *
 *
 *
 *
 * String match as the first example*/
int main(int argc, char **argv) {
//	get parameters from the command line

	//get-opt to get parameters

	////get-opt from mapcg


	//Handle the input data
	//get input from data file and copy the data into host memory
	//make the raw input data fit the scheduler
	string keyword;
	keyword="hello";

	char* inputbuf=new char[11];
	inputbuf="hello\0you\0";
	int* offset;
	offset=new int[2];
    offset[0]=0;
    offset[1]=6;


	//initialize the Specs from the command line
    Specs SchedulerSpecs(argc,argv); //	parse the parameter and store them into Scheduler
    SchedulerSpecs.input= (unsigned int*)offset;
    SchedulerSpecs.input_size=11*sizeof(int);


    //DoLog("hello");

    global_data_t* gbtmp;
    gbtmp->content=inputbuf;
    SchedulerSpecs.gbdata=gbtmp;

	//init the Scheduler
	TaskScheduler MapDSWScheduler;
	MapDSWScheduler.init(&SchedulerSpecs);

    //start timer
	//Start Map-Reduce
	MapDSWScheduler.doMapReduce();
    //calculate the running time of the MapReduce Job

	//get output from the Scheduler output queue


}
