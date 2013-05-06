/*
 * main.cpp
 *
 *  Created on: 2013-4-3
 *      Author: Shiwei Dong
 */
#include <iostream>
#include <fstream>
#include <stdlib.h>
//#include <string>
#include "time.h"
#include "math.h"
#include <vector>
#include <assert.h>
using namespace std;

#include <string.h>
#include "MRLib/TaskScheduler.h"
#include "MRLib/Common.h"
#include "UtilLib/CommonUtil.h"

void map_file(const char * filename, void * & buf, unsigned int & size){
	int fp=open(filename,O_RDONLY);
	if(fp){
		struct stat filestat;
		fstat(fp, &filestat);
		size=filestat.st_size;
		buf=mmap(0,size,PROT_READ,MAP_PRIVATE,fp,0);
	}
	else
		buf=0;
}

/**
 * String match as the first example
 */
int main(int argc, char **argv) {
//	get parameters from the command line

//get-opt to get parameters

////get-opt from mapcg

//Handle the input data
//get input from data file and copy the data into host memory
//make the raw input data fit the scheduler
	string keyword;
	keyword = "hello";

	string filename;
	if (!get_opt(argc, argv, "f", filename)) {
		cout << "usage: " << argv[0] << " -f filename" << endl;
		return 1;
	}

	void * rawbuf;
	unsigned int size;
	map_file(filename.c_str(), rawbuf, size);
	if (!rawbuf) {
		cout << "error opening file " << filename << endl;
		return 1;
	}
	char * filebuf = new char[size];
	memcpy(filebuf, rawbuf, size);

	unsigned int* offsets;
	offsets= new unsigned int[100000];
	int i=0;
	unsigned int offset=0;

	FILE *fp = fopen(filename.c_str(), "r");

	char buf[1024];
	memset(buf, '\0', 1024);

	while (fgets(buf, 1024, fp) != NULL) {
		offsets[i]=offset;
	    offset += (unsigned int)strlen(buf);
		filebuf[offset - 1] = '\0';
		memset(buf, '\0', 1024);
		i++;
	}


//	char* inputbuf = new char[16];
//	inputbuf = "hello\0you\0hello\0";
//	unsigned int* offset;
//	offset = new unsigned int[3];
//	offset[0] = 0;
//	offset[1] = 6;
//	offset[2] = 10;

	//initialize the Specs from the command line
	Specs SchedulerSpecs(argc, argv); //	parse the parameter and store them into Scheduler
	SchedulerSpecs.input = offsets;
	SchedulerSpecs.input_size = i * sizeof(int);

	cout<<"lines"<<i<<endl;
	SchedulerSpecs.printArg();

	global_data_t gbtmp;
	gbtmp.content = filebuf;
	//gbtmp.keyword = "hello";
	SchedulerSpecs.gbdata = &gbtmp;
	SchedulerSpecs.gbdata_size = size;

//	cout << SchedulerSpecs.input[1]<<endl;
	//init the Scheduler

	TaskScheduler MapDSWScheduler;
	MapDSWScheduler.init(&SchedulerSpecs);

	//start timer
	//Start Map-Reduce
	cout << "Starting the Map-Reduce Procedure..." << endl;
	double t1=get_time();
	MapDSWScheduler.doMapReduce();
	double t2=get_time();
	cout<<"==== total time: "<<t2-t1<<endl;

	Output output= MapDSWScheduler.getOutput();
	cout<<"number of output: "<<output.count<<endl;
	for(int i=0;i<output.count && i<100; i++){
		unsigned int keyIdx=output.key_index[i];
		//int keySize=output.index[i].y;
		unsigned int valIdx=output.val_index[i];
		char * link=filebuf+*(unsigned int*)((char*)(output.output_keys)+keyIdx);
		//cout<<"valueId"<<valIdx<<endl;
		int val=*(unsigned int*)((char*)(output.output_vals)+valIdx);
		cout<<"("<<val<<")"<<link<<endl;
	}
	/*
	 //calculate the running time of the MapReduce Job

	 //get output from the Scheduler output queue

	 */
	delete[] filebuf;
	return 0;
}
