/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * TaskScheduler.cu
 *
 *  Created on: 2013-4-3
 *      Author: Shiwei Dong
 */

#include "TaskScheduler.h"
#include "MemAlloc.h"
#include "SMCache.h"
#include "Intermediate.h"
#include "Common.h"
#include "../UtilLib/CommonUtil.h"
#include "../UtilLib/GpuUtil.h"
#include <iostream>
using namespace std;

//===========================
//  GPU device functions
//===========================
__device__ void emit_intermediate(Intermediate* inter, SMCache* Cache,
		MemAlloc* Mem_Alloc) {
	//if Cache mode then use Cache else insert into Mem_Alloc
	//if(threadIdx.x==0){printf("aaaa%d\n\n",inter->keysize);}
	Cache->insert(inter, Mem_Alloc);
	printf("11111111111\n\n");
}

#include "../UserDef/Mapreduce.h"

//============================
// All the GPU functions
//============================
//kernel launched on host
/**
 * Mapper function:
 *
 */
__global__ void Mapreducer(void* global_data_d, unsigned int* input_offset_d,
		unsigned int* input_size_d, MemAlloc* mem_alloc_d) {

	//global thread id
	unsigned int threadID = getThreadID();
	unsigned int Num_threads = getNumThreads();

	//thread id etc in the block
//	unsigned int bid=blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int Num_threads_b = blockDim.x;

	//setup the memory allocator
	mem_alloc_d->Start_MA_kernal();
	__syncthreads();

	//init the shared memory Cache
	__shared__ SMCache Cache[CACHEGROUP];

	//the first thread of each group initialize the SMCache using different caching mode

	unsigned int threadsPerGroup = align(Num_threads_b, CACHEGROUP) / CACHEGROUP;
	unsigned int gid = tid / threadsPerGroup;

	//initialize the global domerge sign
	domerge = 0;

	if (tid % threadsPerGroup == 0) {
		Cache[gid].init();
	}
	__syncthreads();

	//divide input data into chunks, each thread deal with up to chunks/Num_threads map job
	unsigned int Num_inputs = *input_size_d / sizeof(int);
//	if(threadID==0){
//	printf("Num_inputs=%d\n",Num_inputs);
//	}

//may have problem here
	for (int i = threadID; i < Num_inputs; i += Num_threads) {
		//do the map job by calling the user defined map function, get the key and value and insert them into Cache
		//the input_offset_d[0] is the first offset in the global memory, since it is only a job, so the offset should all minus input_offset_d[0]
		//printf("%s\n", global_data_d);
		map(global_data_d, input_offset_d[i] - input_offset_d[0], &Cache[gid],
				mem_alloc_d);
		printf("2222222222222\n\n");
	}

	//merge all the existing cache into the global device memory
	__syncthreads();
	printf("3333333\n\n");
	mem_alloc_d->Merge_SMCache(&Cache[gid]);

}

/**
 * The main entrance of the scheduler
 */
void TaskScheduler::doMapReduce() {

	//Slice the input data into pieces which is maintained as Jobs in a Job sequence
	slice();

	//while the Jobqueue is not empty, Pop out a job from the Job sequence and then do the Map job.
	while (!JobQueue.empty()) {
		Job currentJob = JobQueue.front();
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
void TaskScheduler::slice() {
	//to be done
	Job job;
	job.input = mySpecs->input;
	job.input_size = mySpecs->input_size;
	//job.unit_size=mySpecs->unit_size;
	job.data_size = mySpecs->gbdata_size;
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
void TaskScheduler::doMap(const Job* job) {

	//1.Initialize the memory allocator on the GPU
	//to be done ,initialize all the memory allocator to 0
	MemAlloc* mem_alloc_d;
//	cout<<sizeof(MemAlloc)<<endl;
	CE(cudaMalloc(&mem_alloc_d, sizeof(MemAlloc)));
	CE(cudaMemset(mem_alloc_d, 0, sizeof(MemAlloc)));

	//2.malloc space for the Job data and job input offsets on device
	void* global_data_d;
	unsigned int* input_offset_d;
	unsigned int* input_size_d;

	//data_size is the size of global_data_d
	unsigned int data_size = job->data_size;

	//may have problem job->input[0]?
//cout<<data_size;
//cout<<job->input[0]<<endl;
//	cout<<mySpecs->gbdata->keyword;
	global_data_t* job_data_g = (global_data_t*) (mySpecs->gbdata
			+ job->input[0]);

	CE(cudaMalloc(&global_data_d, data_size));
	CE(cudaMalloc(&input_offset_d,job->input_size));
	CE(cudaMalloc(&input_size_d,sizeof(unsigned int)));

	//3.copy the Job data and additional information into the GPU device memory
	memcpyH2D(global_data_d, job_data_g->content, data_size);
	memcpyH2D(input_offset_d, job->input, job->input_size);
	memcpyH2D(input_size_d, &job->input_size, sizeof(unsigned int));

	//4.start the Map job by launching a GPU kernel function
	dim3 Grid(mySpecs->dim_grid, 1, 1);
	dim3 Block(mySpecs->dim_block, 1, 1);
	//cout<<data_size;
	//cout<<"aaa"<<job_data_g->keyword<<endl;
	Mapreducer<<<Grid, Block>>>(global_data_d, input_offset_d, input_size_d,
			mem_alloc_d);
	//cudaThreadExit();
	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Mapreducer error: ");
	//cout<<"ok";
	//5.Collect output from the device memory
	//copy the memory allocator back to the host memory and merge it into an ouput array

	//free used data pointers
	CE(cudaFree(global_data_d));
	CE(cudaFree(input_offset_d));
	CE(cudaFree(input_size_d));
}

void TaskScheduler::doReduce() {

}

/**
 * Construction function and initialize function
 */
//initialize the TaskScheduler by passing the Specs to it
void TaskScheduler::init(const Specs* specs_in) {
	mySpecs = specs_in;
}

TaskScheduler::TaskScheduler(const Specs* specs_in) {
	init(specs_in);
}

TaskScheduler::TaskScheduler() {

}
