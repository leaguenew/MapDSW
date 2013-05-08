/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * TaskScheduler.h
 *
 *  Created on: 2013-4-3
 *      Author: Shiwei Dong
 */

#ifndef TASKSCHEDULER_H_
#define TASKSCHEDULER_H_

#include <queue>

struct Specs;
struct Job;
struct Output;


/**
 * TaskScheduler is a class wit
 */
class TaskScheduler {
public:
	//Interface
	TaskScheduler();
	TaskScheduler(const Specs*);
	void init(const Specs*);
	void destroy();

	void doMapReduce();
	//to be improved
	Output getOutput();

private:
	//slice the input data into job queue to prevent overflow
	void slice();
	//schedule map
	void doMap(const Job*);
	//schedule reduce
//	void doReduce();

	//void mergeOutput();  multiple threads

private:
	const Specs* mySpecs;
	std::queue<Job> JobQueue;
	std::queue<Output> OutputQueue;

};


#endif /* TASKSCHEDULER_H_ */
