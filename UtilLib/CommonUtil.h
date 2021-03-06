/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * CommonUtil.h
 *
 *  Created on: 2013-4-17
 *      Author: Shiwei Dong
 */

#ifndef COMMONUTIL_H_
#define COMMONUTIL_H_


#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <sstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <cuda.h>



//=====================================
// parse command line option
//=====================================
template<class T>
inline bool get_opt(int argc, char * argv[], const char * option, T & output){
	using namespace std;
	bool opt_found=false;
	int i;
	for(i=0;i<argc;i++){
		if(argv[i][0]=='-'){
			string str(argv[i]+1);
			for(int j=0;j<str.size();j++)
				str[j]=tolower(str[j]);
			string opt(option);
			for(int j=0;j<opt.size();j++)
				opt[j]=tolower(opt[j]);
			if(str==opt){
				opt_found=true;
				break;
			}
		}
	}
	if(opt_found){
		istringstream ss(argv[i+1]);
		ss>>output;
	}
	return opt_found;
}

inline double get_time(){
	//cudaThreadSynchronize();
	timeval t;
	gettimeofday(&t, NULL);
	return (double)t.tv_sec+(double)t.tv_usec/1000000;
}


#endif /* COMMONUTIL_H_ */
