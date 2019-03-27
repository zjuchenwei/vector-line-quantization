#ifndef NEARESTNEIGHBOR_HELPER_H
#define NEARESTNEIGHBOR_HELPER_H

/*! \file  helper.hh
    \brief a collection of helper classes
 */
//#define OUTPUT

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

using namespace std;

#define MAX_THREADS 512
#define MAX_BLOCKS 65535
#define WARP_SIZE 32
namespace faiss { namespace gpu {


void outputMat(const std::string& _S, const float* _A,
		uint _rows, uint _cols,cudaStream_t stream);

void outputVec(const std::string& _S, const float* _v,
		uint _n,cudaStream_t stream);

void outputVecChar(const std::string& _S, const char* _v,
		uint _n,cudaStream_t stream);
void outputVecUint8(const std::string& _S, const uint8_t* _v,
		uint _n,cudaStream_t stream);
void outputVecUint(const std::string& _S, const uint* _v,
		uint _n,cudaStream_t stream);
void outputVecUShort(const std::string& _S, const ushort* _v,
		uint _n,cudaStream_t stream);

void outputVecInt(const std::string& _S, const int* _v,uint _n,cudaStream_t stream);

void outputVecLong(const std::string& _S, const long* _v,uint _n,cudaStream_t stream);

void checkPrefixSumOffsets(const int* _v,uint _n,cudaStream_t stream);


}

} /* namespace */



#endif /* NEARESTNEIGHBOR_HELPER_H */
