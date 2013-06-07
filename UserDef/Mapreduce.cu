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

/**
 * String match
 */
__device__ void map(void* global_data_d, unsigned int offset,
		MemAlloc* Mem_Alloc, SMCache* Cache, bool *isFail) {

	char* keyword = "dahz\0";
	char* line = (char*) global_data_d + offset;

	//store the position of the key, emit as the value, line as the key
	char* pos = line;

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
					//int length=curr-pos;
					if ((*curr == ' ' || *curr == '\0')) {
						//the intermediate is the line of the +input_offset_d[0]
						Intermediate inter;
						inter.init(&offset, sizeof(int), &pos_line,
								sizeof(int));

						emit_intermediate(&inter, Mem_Alloc, Cache, isFail);
					}

					pos = curr;
				}

			} else {
				pos++;
				break;
			}
		}
	}

}

__device__ void reduce(Intermediate* current, Intermediate* inter) {

}

/**
 * compare the value, Most time, this function does not need to be changed
 */
__device__ int compare(const void* a, const void* b, unsigned short size_a,
		unsigned short size_b) {

	//equal: 0 smaller than -1 bigger than 1
	/*unsigned int A, B;
	 copyVal(&A, (void*) a, sizeof(unsigned int));
	 copyVal(&B, (void*) b, sizeof(unsigned int));

	 if (A > B)
	 return 1;
	 else if (A < B)
	 return -1;
	 else
	 return 0;*/

	return -2;
}

