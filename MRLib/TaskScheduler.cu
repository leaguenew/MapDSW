/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * TaskScheduler.cu
 *
 *  Created on: 2013-4-3
 *      Author: Shiwei Dong
 */

#include <cuda.h>
#include <stdio.h>
#include "TaskScheduler.h"
#include "MemAlloc.h"
#include "SMCache.h"
#include "../UtilLib/CommonUtil.h"
#include "../UtilLib/GpuUtil.h"
#include "../UserDef/Mapreduce.cu"


//============================
// All the GPU functions
//============================
//kernel launched on host
/**
 * Mapper function:
 *
 */
__global__ void Mapper (MemAlloc* mem_alloc_d){

	//global thread id
	unsigned int threadID=getThreadID();
	unsigned int Num_threads=getNumThreads();

	//thread id etc in the block
	unsigned int bid=blockIdx.x;
	unsigned int tid=threadIdx.x;
	unsigned int Num_threads_b=blockDim.x;

	//setup the memory allocator
	mem_alloc_d->Start_MA_kernal();

	//init the shared memory Cache
	__shared__ SMCache Cache[CACHEGROUP];

	//the first thread of each group initialize the SMCache using different caching mode
	unsigned int threadsPerGroup = align(Num_threads_b,CACHEGROUP)/CACHEGROUP;
	unsigned int gid=tid/threadsPerGroup;

	if(tid%threadsPerGroup==0){
		Cache[gid].init(mode);
	}
    __syncthreads();

    //divide input data into chunks in order to deal with big input
    ////////////////to be done
    unsigned int chunks=;
    for(int i=threadID;i<=Num_threads;i+=chunks){
    	//do the map job by calling the user defined map function, get the key and value and insert them into Cache
    	Intermediate *inter= map(offset,Cache[gid]);
    	Cache[gid].insert(inter, mem_alloc_d);
    }

	//

}


//===========================
//  GPU device functions
//===========================
__device__ Intermediate* emit_intermediate(){
	return;
}

/**
 * The main entrance of the scheduler
 */
void TaskScheduler::doMapReduce(){

	//Slice the input data into pieces which is maintained as Jobs in a Job sequence
	slice();

	//while the Jobqueue is not empty, Pop out a job from the Job sequence and then do the Map job.
	while(!JobQueue.empty()){
		Job currentJob=JobQueue.front();
		JobQueue.pop();
		doMap(&currentJob);
	}

	//when all the map job was over, the GPU sync the result.

	// Do the Reduce job
	doReduce();
	// merge the output from the Reduce stage and finish the Mapreduce job

}


/**
 * Slice the input data into pieces which is maintained as Jobs in a Job sequence
 */
void TaskScheduler::slice(){
	//to be done
	Job job;
	job.input=mySpecs->input;
	job.input_size=mySpecs->input_size;
	//job.unit_size=mySpecs->unit_size;
	job.data_size=mySpecs->gbdata_size;
	JobQueue.push(job);
}



/**
 * Do the map work for the current Job
 * 1.Initialize the dynamic memory allocator on the GPU
 * 2.malloc space for the Job data
 * 3.copy the input data and additional information into the GPU device memory
 * 4.start the Map job by lauching a GPU kernel function
 * 5.Collect output from the device memory
 */
void TaskScheduler::doMap(const Job* job){

	//1.Initialize the memory allocator on the GPU
	MemAlloc* mem_alloc_d;
	CE(cudaMalloc(&mem_alloc_d, sizeof(MemAlloc)));

	//2.malloc space for the Job data and job input offsets on device
	unsigned int data_size=job->data_size;
	global_data_t* job_data_g=mySpecs->gbdata+job->input[0];
	CE(cudaMalloc(&global_data_d, data_size));
	CE(cudaMalloc(&input_offset_d,job->input_size));
	CE(cudaMalloc(&input_size_d,sizeof(int)));

	//3.copy the Job data and additional information into the GPU device memory
	memcpyH2D(global_data_d ,job_data_g, data_size);
	memcpyH2D(input_offset_d,job->input,job->input_size);
	memcpyH2D(input_size_d,&job->input_size,sizeof(int));

	//4.start the Map job by launching a GPU kernel function
	dim3 Grid(mySpecs->dim_grid,1,1);
	dim3 Block(mySpecs->dim_block,1,1);
	Mapper<<<Grid,Block>>>(mem_alloc_d);

	//5.Collect output from the device memory

	//free used data pointers
	CE(cudaFree(global_data_d));
	CE(cudaFree(input_offset_d));
}


void TaskScheduler::doReduce(){

}


/**
 * Construction function and initialize function
 */
//initialize the TaskScheduler by passing the Specs to it
void TaskScheduler::init(const Specs* specs_in){
	mySpecs=specs_in;
}
