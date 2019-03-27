#include "./helper.cuh"
#define OUTPUT
namespace faiss { namespace gpu {

__global__ void outputMatKernel(const float*_A, uint _rows,
		uint _cols) {
	if (threadIdx.x == 0) {
		for (int j = 0; j < _rows; j++) {
			for (int i = 0; i < _cols; i++)
				printf("%.4f ", _A[j * _cols + i]);
			printf("\n");
		}

	}
}

void outputMat(const std::string& _S, const float* _A,
		uint _rows, uint _cols,cudaStream_t stream) {
#ifndef OUTPUT
	return;
#endif
	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputMatKernel<<<grid, block,0,stream>>>(_A, _rows, _cols);



}

__global__ void outputVecCharKernel(const char* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n; i++)
			printf("%5d ", (uint)_v[i]);
		printf("\n");

	}
}

void outputVecChar(const std::string& _S, const char* _v,
		uint _n,cudaStream_t stream) {
#ifndef OUTPUT
	return;
#endif

	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecCharKernel<<<grid, block,0,stream>>>(_v, _n);



}

__global__ void outputVecKernel(const float* _v, uint _n) {
//printf("asdfasdfas\n");
	if (threadIdx.x == 0) {

		for (int i = 0; i < _n; i++)
			printf("%.4f ", _v[i]);
		printf("\n");

	}
}

void outputVec(const std::string& _S, const float* _v,
		uint _n,cudaStream_t stream) {
#ifndef OUTPUT
	return;
#endif

	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecKernel<<<grid, block,0,stream>>>(_v, _n);



}


__global__ void outputVecUIntKernel(const uint* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n; i++)
			printf("%5d ", _v[i]);
		printf("\n");

	}
}

void outputVecUint(const std::string& _S, const uint* _v,
		uint _n,cudaStream_t stream) {
#ifndef OUTPUT
	return;
#endif

	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecUIntKernel<<<grid, block,0,stream>>>(_v, _n);


}

__global__ void outputVecUInt8Kernel(const uint8_t* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n; i++)
			printf("%5d ", _v[i]);
		printf("\n");

	}
}

void outputVecUint8(const std::string& _S, const uint8_t* _v,
		uint _n,cudaStream_t stream) {
#ifndef OUTPUT
	return;
#endif

	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecUInt8Kernel<<<grid, block,0,stream>>>(_v, _n);

}
__global__ void outputVecUShortKernel(const ushort* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n; i++)
			printf("%5d ", _v[i]);
		printf("\n");

	}
}
void outputVecUShort(const std::string& _S, const ushort* _v,
		uint _n,cudaStream_t stream) {
#ifndef OUTPUT
	return;
#endif

	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecUShortKernel<<<grid, block,0,stream>>>(_v, _n);
}

__global__ void outputVecIntKernel(const int* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n; i++)
			printf("%5d ", _v[i]);
		printf("\n");

	}
}

void outputVecInt(const std::string& _S, const int* _v,
		uint _n,cudaStream_t stream) {
#ifndef OUTPUT
	return;
#endif
	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecIntKernel<<<grid, block,0,stream>>>(_v, _n);


}

__global__ void checkPrefixSumOffsetsKernel(const int* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n-1; i++){
		    if(_v[i]>_v[i+1]){
		        printf("_v[%d]:%5d,_v[%d]:%5d\n", i,_v[i],i+1,_v[i+1]);
		        break;
		    }

		}

     }
}
void checkPrefixSumOffsets(const int* _v,uint _n,cudaStream_t stream) {
#ifndef OUTPUT
	return;
#endif
	cout << "PrefixSumOffsets" << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	checkPrefixSumOffsetsKernel<<<grid, block,0,stream>>>(_v, _n);


}

__global__ void outputVecLongKernel(const long* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n; i++)
			printf("%ld ", _v[i]);
		printf("\n");

	}
}

void outputVecLong(const std::string& _S, const long* _v,uint _n,cudaStream_t stream)
{
#ifndef OUTPUT
	return;
#endif
	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecLongKernel<<<grid, block,0,stream>>>(_v, _n);

}



}
} /* namespace */
