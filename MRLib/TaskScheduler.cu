/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * TaskScheduler.cu
 *
 *  Created on: 2013-4-3
 *      Author: Shiwei Dong
 */

#include <stdio.h>
#include "TaskScheduler.h"
#include "../UtilLib/CommonUtil.h"
/**
 * All the GPU functions
 */
//kernel launched on host
__global__ void Mapper (){
	//get some memory space from the dynamic memory allocator to store the intermediate keys and values

	//if selected to use SMCache
	//initialize the Reduction Object on shared memory(be aware the the shared memory is limited)
	//do the map job by calling the user defined map function

	//
	//

}


/*===========================
 * GPU device functions
 *===========================*/
__device__ void emit_intermediate(){

}

/**
 * The main entrance of the scheduler
 */
void TaskScheduler::doMapReduce(){

	//Slice the input data into pieces which is maintained as Jobs in a Job sequence
	slice();

	//while the Jobqueue is not empty, Pop out a job from the Job sequence and then do the Map job.
	//Do the Map
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
	job.input=mySpecs->offsets;
	job.input_size=mySpecs->input_size;
	JobQueue.push(job);
}



/**
 * Do the map work for the current Job
 * 1.Initialize the dynamic memory allocator on the GPU
 * 2.malloc enough space for the input data using the dynamic memory allocator
 * 3.copy the input data and additional information into the GPU device memory
 * 4.start the Map job by lauching a GPU kernel function
 * 5.Collect output from the device memory
 */
void TaskScheduler::doMap(const Job* job){

	//1.Initialize the dynamic memory allocator on the GPU
	//malloc enough memory space for the memory pool in the GPU device memory and set the memory space to be unused

	//2.malloc enough space for the input data using the dynamic memory allocator

	//3.copy the Job data and additional information into the GPU device memory

	//4.start the Map job by launching a GPU kernel function
	Mapper<<<grid,block>>>();

	//finish the map Job and store intermediate data in the shared memory

	//make sure all the result has been efficiently merged and stored in the device memory.

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
