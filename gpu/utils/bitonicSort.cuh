#ifndef BITONIC_SORT_CUH
#define BITONIC_SORT_CUH

#include <stdio.h>
#include <stdlib.h>

namespace faiss { namespace gpu {
template<class T>
__device__ void swap1(T& _a, T&_b) {
	T h = _a;
	_a = _b;
	_b = h;
}

// parallel bitonic sort
template<class T>
__device__ void bitonic3(volatile T _val[], volatile uint _idx[], uint _N) {

	for (int k = 2; k <= _N; k <<= 1) {

		// bitonic merge
		for (int j = k / 2; j > 0; j /= 2) {
			int ixj = threadIdx.x ^ j;  // XOR
			if ((ixj > threadIdx.x) && (ixj < _N)) {
				if ((threadIdx.x & k) == 0) // ascending - descending
						{
					if (_val[threadIdx.x] > _val[ixj]) {

						swap1(_val[threadIdx.x], _val[ixj]);
						swap1(_idx[threadIdx.x], _idx[ixj]);
					}
				} else {
					if (_val[threadIdx.x] < _val[ixj]) {

						swap1(_val[threadIdx.x], _val[ixj]);
						swap1(_idx[threadIdx.x], _idx[ixj]);
					}

				}
			}
			__syncthreads();
		}
	}
}


// parallel bitonic sort
template<class T>
__device__ void bitonicLarge(volatile T _val[], volatile uint _idx[], uint _N) {

	for (int k = 2; k <= _N; k <<= 1) {

		// bitonic merge
		for (int j = k / 2; j > 0; j /= 2) {

			for (int tid = threadIdx.x; tid < _N; tid += blockDim.x) {
				int ixj = tid ^ j;  // XOR
				if ((ixj > tid) && (ixj < _N)) {
					if ((tid & k) == 0) // ascending - descending
							{
						if (_val[tid] > _val[ixj]) {

							swap1(_val[tid], _val[ixj]);
							swap1(_idx[tid], _idx[ixj]);
						}
					} else {
						if (_val[tid] < _val[ixj]) {

							swap1(_val[tid], _val[ixj]);
							swap1(_idx[tid], _idx[ixj]);
						}

					}
				}
			}
			__syncthreads();
		}
	}
}

// parallel bitonic sort (descending)
template<class T>
__device__ void bitonic3Descending(volatile T _val[], volatile uint _idx[],
		uint _N) {

	for (int k = 2; k <= _N; k <<= 1) {

		// bitonic merge
		for (int j = k / 2; j > 0; j /= 2) {
			int ixj = threadIdx.x ^ j;  // XOR
			if ((ixj > threadIdx.x) && (ixj < _N)) {
				if ((threadIdx.x & k) != 0) // ascending - descending
						{
					if (_val[threadIdx.x] > _val[ixj]) {

						swap1(_val[threadIdx.x], _val[ixj]);
						swap1(_idx[threadIdx.x], _idx[ixj]);
					}
				} else {
					if (_val[threadIdx.x] < _val[ixj]) {

						swap1(_val[threadIdx.x], _val[ixj]);
						swap1(_idx[threadIdx.x], _idx[ixj]);
					}

				}
			}
			__syncthreads();
		}
	}
}

}};
#endif
