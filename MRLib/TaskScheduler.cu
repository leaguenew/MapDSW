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
//#include "../UtilLib/CommonUtil.h"
#include "../UtilLib/GpuUtil.h"
#include "../UtilLib/Scan.h"
#include <iostream>
using namespace std;

//===========================
//  GPU device functions
//===========================
__device__ void emit_intermediate(Intermediate* inter, SMCache* Cache,
		MemAlloc* Mem_Alloc) {
	//if Cache mode then use Cache else insert into Mem_Alloc
	//printf("emitting an intermediate in thread %d\n", threadIdx.x);
	Cache->insert(inter, Mem_Alloc);
}

#include "../UserDef/Mapreduce.h"

//============================
// All the GPU functions
//============================
//kernel launched on host

__global__ void Mapreducer(void* global_data_d, unsigned int* input_offset_d,
		unsigned int* input_size_d, MemAlloc* mem_alloc_d) {

	//global thread id
	unsigned int threadID = getThreadID();
	unsigned int Num_threads = getNumThreads();

	//thread id etc in the block
	unsigned int bid=blockIdx.x;
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
	//make sure that all the threads enters the for loop at the same time even though some do not do
	//the map job, since they may have to synchronize in the loop
	unsigned int Num_chunks = align(Num_inputs, Num_threads);

//	if (tid == 0)
//		printf("Num_chunks %d Num_threads %d\n", Num_chunks, Num_threads);
	for (int i = threadID; i < Num_chunks; i += Num_threads) {
		//do the map job by calling the user defined map function, get the key and value and insert them into Cache
		//the input_offset_d[0] is the first offset in the global memory, since it is only a job, so the offset should all minus input_offset_d[0]

		if (i < Num_inputs) {
			map(global_data_d, input_offset_d[i] - input_offset_d[0],
					&Cache[gid], mem_alloc_d);
			//printf("Map done in %d\n", threadID);
			/**
			 * if the SMCache is full, stop all the threads and then swap the SMCache out and merge to the
			 * memory allocator. Then flush the SMCache, and map again
			 */
			if (domerge) {
				if (tid == 0)
					printf("merge!! this block\n");
				__syncthreads();
				mem_alloc_d->Merge_SMCache(&Cache[gid], gid);
				__syncthreads();
				if (tid % threadsPerGroup == 0) {
					Cache[gid].flush();
				}
				if (tid == 0) {
					atomicExch(&domerge, 0);
				}

				map(global_data_d, input_offset_d[i] - input_offset_d[0],
						&Cache[gid], mem_alloc_d);

			}
		}

	}

	//merge all the existing cache into the global device memory
	__syncthreads();
	//if (tid == 0)
		//printf("merging into the mem alloc\n");

	mem_alloc_d->Merge_SMCache(&Cache[gid], gid);

//	if(tid==0&&bid==0){
//		for(int i=0;i<MEM_BUCKETS;i++){
//			if(mem_alloc_d->value_index[i]!=0){
//				if(*(unsigned int*)(mem_alloc_d->getaddress(mem_alloc_d->value_index[i]))>10000)
//				printf("value: %d\n",*(unsigned int*)(mem_alloc_d->getaddress(mem_alloc_d->value_index[i])));
//			}
//		}
//	}

}

__global__ void copy_hash_to_array(MemAlloc *mem_alloc_d, void *key_array,
		unsigned int *key_start_per_bucket, void *val_array,
		unsigned int *val_start_per_bucket, unsigned int *pair_start_per_bucket,
		unsigned int *key_index, unsigned int *val_index) {

	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;
	const unsigned int num_threads = gridDim.x * blockDim.x;

	for (int i = bid * blockDim.x + tid; i < MEM_BUCKETS; i += num_threads) {
		if ((mem_alloc_d->key_size)[i] != 0) {
			int key_size = (int) mem_alloc_d->key_size[i];
			int val_size = (int) mem_alloc_d->value_size[i];
			void *key = mem_alloc_d->getaddress(mem_alloc_d->key_index[i]);
			void *val = mem_alloc_d->getaddress(mem_alloc_d->value_index[i]);
			unsigned int key_array_start = key_start_per_bucket[i];
			unsigned int val_array_start = val_start_per_bucket[i];
			unsigned int offset_pos = pair_start_per_bucket[i];
			copyVal((char *) key_array + key_array_start, key, key_size);
			copyVal((char *) val_array + val_array_start, val, val_size);
			key_index[offset_pos] = key_array_start;
			val_index[offset_pos] = val_array_start;
		}
	}
}

/**
 * The main entrance of the scheduler
 */
void TaskScheduler::doMapReduce() {

	//Slice the input data into pieces which is maintained as Jobs in a Job sequence
	slice();

	//while the Jobqueue is not empty, Pop out a job from the Job sequence and then do the Map job.
	while (!JobQueue.empty()) {
		cout << "A Job has been initiated..." << endl;
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

	CE(cudaMalloc(&mem_alloc_d, sizeof(MemAlloc)));
	CE(cudaMemset(mem_alloc_d, 0, sizeof(MemAlloc)));

	//2.malloc space for the Job data and job input offsets on device
	void* global_data_d;
	unsigned int* input_offset_d;
	unsigned int* input_size_d;

	//data_size is the size of global_data_d
	unsigned int data_size = job->data_size;

	//may have problem job->input[0]?

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

	Mapreducer<<<Grid, Block>>>(global_data_d, input_offset_d, input_size_d,
			mem_alloc_d);

	cudaThreadSynchronize();
	CUT_CHECK_ERROR("Mapreducer error: ");
	cout << "The Map-Reduce procedure on GPU succeeded" << endl;

	//free used data pointers
	CE(cudaFree(global_data_d));
	CE(cudaFree(input_offset_d));
	CE(cudaFree(input_size_d));

	//5.Collect output from the device memory
	//copy the memory allocator back to the host memory and merge it into an ouput array

	cout << "Start collecting results from GPU" << endl;

	//first allocate exact memory space to the output array on GPU by calculating the prefix sum

	unsigned int * key_start_per_bucket;
	unsigned int * val_start_per_bucket;
	unsigned int * pair_start_per_bucket;

	cudaMalloc((void **) &key_start_per_bucket,
			sizeof(unsigned int) * MEM_BUCKETS);
	cudaMalloc((void **) &val_start_per_bucket,
			sizeof(unsigned int) * MEM_BUCKETS);
	cudaMalloc((void **) &pair_start_per_bucket,
			sizeof(unsigned int) * MEM_BUCKETS);

	cudaMemset(key_start_per_bucket, 0, sizeof(unsigned int) * MEM_BUCKETS);
	cudaMemset(val_start_per_bucket, 0, sizeof(unsigned int) * MEM_BUCKETS);
	cudaMemset(pair_start_per_bucket, 0, sizeof(unsigned int) * MEM_BUCKETS);

	unsigned int * key_size_per_bucket = mem_alloc_d->key_size;
	unsigned int * val_size_per_bucket = mem_alloc_d->value_size;
	unsigned int * pairs_per_bucket = mem_alloc_d->used;

	int total_key_size = prefix_sum(key_size_per_bucket, key_start_per_bucket,
			MEM_BUCKETS);
	int total_value_num = prefix_sum(pairs_per_bucket, pair_start_per_bucket,
			MEM_BUCKETS);
	int total_value_size = prefix_sum(val_size_per_bucket, val_start_per_bucket,
			MEM_BUCKETS);
	int total_key_num = total_value_num;
	cout << "total key size: " << total_key_size << endl;
	cout << "total value size: " << total_value_size << endl;
	cout << "total key_num: " << total_key_num << endl;
	cout << "total value_num: " << total_value_num << endl;

	//copy back to the host memory from the device memory
	char *output_keys_d;
	char *output_vals_d;
	unsigned int *key_index_d;
	unsigned int *val_index_d;

	cudaMalloc((void **) &output_keys_d, total_key_size);
	cudaMalloc((void **) &output_vals_d, total_value_size);
	cudaMalloc((void **) &key_index_d, sizeof(unsigned int) * total_key_num);
	cudaMalloc((void **) &val_index_d, sizeof(unsigned int) * total_value_num);

	copy_hash_to_array<<<Grid, Block>>>(mem_alloc_d, output_keys_d,
			key_start_per_bucket, output_vals_d, val_start_per_bucket,
			pair_start_per_bucket, key_index_d, val_index_d);
	cudaThreadSynchronize();

	/*Allocate space on host*/
	char *output_keys = (char *) malloc(total_key_size);
	char *output_vals = (char *) malloc(total_value_size);
	unsigned int *key_index = (unsigned int *) malloc(
			sizeof(unsigned int) * total_key_num);
	unsigned int *val_index = (unsigned int *) malloc(
			sizeof(unsigned int) * total_value_num);

	memcpyD2H(output_keys, output_keys_d, total_key_size);
	memcpyD2H(output_vals, output_vals_d, total_value_size);
	memcpyD2H(key_index, key_index_d, sizeof(unsigned int) * total_key_num);
	memcpyD2H(val_index, val_index_d, sizeof(unsigned int) * total_value_num);

	//store the output
	Output output;
	output.output_keys = output_keys;
	output.output_vals = output_vals;
	output.key_index = key_index;
	output.val_index = val_index;
	output.count = total_key_num;

	OutputQueue.push(output);

	cudaFree(key_start_per_bucket);
	cudaFree(val_start_per_bucket);
	cudaFree(pair_start_per_bucket);

	cudaFree(output_keys_d);
	cudaFree(output_vals_d);
	cudaFree(key_index_d);
	cudaFree(val_index_d);

}

void TaskScheduler::doReduce() {

}

//need to be modified
Output TaskScheduler::getOutput() {
	Output tmpoutput;
	tmpoutput = OutputQueue.back();
	return tmpoutput;
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
