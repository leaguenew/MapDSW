/*
 * main.cpp
 *
 *  Created on: 2013-4-3
 *      Author: Shiwei Dong
 */
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "time.h"
#include "math.h"
#include <vector>
#include <assert.h>
#include <string.h>
using namespace std;

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
//get parameters from the command line


//Handle the input data
//get input from data file and copy the data into host memory
//make the raw input data fit the scheduler


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

	cout<<"There are "<<i<<" records"<<endl;
	//initialize the Specs from the command line
	//parse the parameter and store them into Scheduler
	Specs SchedulerSpecs(argc, argv);
	SchedulerSpecs.input = offsets;
	SchedulerSpecs.input_size = i * sizeof(int);

	SchedulerSpecs.printArg();

	global_data_t gbtmp;
	gbtmp.content = filebuf;
	SchedulerSpecs.gbdata = &gbtmp;
	SchedulerSpecs.gbdata_size = size;

	//init the Scheduler
	TaskScheduler MapDSWScheduler;
	MapDSWScheduler.init(&SchedulerSpecs);

	//start timer
	//Start Map-Reduce
	cout << "Starting the Map-Reduce Procedure..." << endl;
	double t1=get_time();
	MapDSWScheduler.doMapReduce();
	double t2=get_time();

	Output output= MapDSWScheduler.getOutput();
	cout<<"number of output: "<<output.count<<endl;
	for(int i=0;i<output.count && i<5; i++){
		unsigned int keyIdx=output.key_index[i];
		unsigned int valIdx=output.val_index[i];
		char * link=filebuf+*(unsigned int*)((char*)(output.output_keys)+keyIdx);
		int val=*(unsigned int*)((char*)(output.output_vals)+valIdx);
		cout<<"("<<val<<")"<<link<<endl;
	}

	//MapDSWScheduler.destroy();

	cout<<"==== total time: "<<t2-t1<<endl;

	delete[] filebuf;
	delete [] offsets;
	return 0;
}
