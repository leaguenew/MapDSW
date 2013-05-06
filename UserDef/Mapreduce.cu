/* MapDSW is a MapReduce Framework which was aimed to fully develop the potential
 * of GPU. It is for an undergraduate graduation thesis at CS/SJTU
 *
 * Mapreduce.cu
 *
 *  Created on: 2013-5-3
 *      Author: Shiwei Dong
 */


#include "Mapreduce.h"
#include "../MRLib/Intermediate.h"
#include "../MRLib/SMCache.h"
#include "../MRLib/MemAlloc.h"
#include "stdio.h"


__device__ void emit_intermediate(Intermediate* inter, SMCache* Cache,
		MemAlloc* Mem_Alloc);

/**
 * String match
 */
__device__ void map(void* global_data_d, unsigned int offset, SMCache* Cache, MemAlloc* Mem_Alloc) {
	char* keyword = "zeh\0";
	char* line =(char* )global_data_d + offset;

	//store the position of the key, emit as the value, line as the key
	char* pos = line;
//	printf("%s\n",line);
	//Match the key word, begin from each line
	while (*pos != '\0') {
		char* pkeyword = keyword;
		char* curr = pos;
		//if the current char in the line matches the first char of the keyword
		//compare the next char until keyword reaches an end. If success emit intermediate, else change move to next char.
		while (*pkeyword != '\0') {
			if (*pkeyword == *curr) {
				pkeyword++;
				curr++;
				if (*pkeyword == '\0') {
					int pos_line = pos - line;
					//the intermediate is the line of the +input_offset_d[0]
					Intermediate inter;
					inter.init(&offset, sizeof(int), &pos_line, sizeof(int));

					emit_intermediate(&inter, Cache, Mem_Alloc);
					pos = curr;
				}

			} else {
				pos++;
				break;
			}
		}
	}
}

__device__ void reduce() {

}

/**
 * compare the value, Most time, this function does not need to be changed
 */
__device__ bool compare(const void* a, const void* b, unsigned short size) {
	char* comA= (char*) a;
	char* comB= (char*) b;
	for(int i=0; i < size; i++){
		if(*comA!=*comB){
			return false;
		}
		comA++;
		comB++;
	}
	//Sprintf("same!!!\n");
	return true;
}

