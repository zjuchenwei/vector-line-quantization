/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include <algorithm>
#include <iostream>
#include "../../FaissAssert.h"
#include "../utils/Limits.cuh"
#include "../utils/Select.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/helper.cuh"
#include "../utils/MathOperators.cuh"
#include "../utils/Tensor.cuh"
#include "../utils/StaticUtils.h"
#include "../utils/bitonicSort.cuh"
#include "../utils/ConversionOperators.cuh"
namespace faiss { namespace gpu {

template <typename T>
__global__ void outputVecTKernel(const T* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n; i++)
			printf("%.4f ", _v[i]);
		printf("\n");

	}
}

template <typename T>
void outputVecT(const std::string& _S, const T* _v,
		uint _n,cudaStream_t stream) {

	std::cout << _S << std::endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecTKernel<T><<<grid, block,0,stream>>>(_v, _n);


}

template <typename T, int kRowsPerBlock, int kRowUnroll, int kColLoad>
__global__ void sumAlongColumns(Tensor<T, 1, true> input,
                                Tensor<T, 2, true> output) {
  static_assert(kRowsPerBlock % kRowUnroll == 0, "must fit rows");

  // blockIdx.x: which chunk of rows we are responsible for updating
  // blockIdx.y: which chunk of columns we are responsible for
  // updating
  int rowStart = blockIdx.x * kRowsPerBlock;
  int rowEnd = rowStart + kRowsPerBlock;
  int colStart = blockIdx.y * blockDim.x * kColLoad;

  // FIXME: if we have exact multiples, don't need this
  bool endRow = (blockIdx.x == gridDim.x - 1);
  bool endCol = (blockIdx.y == gridDim.y - 1);

  if (endRow) {
    if (output.getSize(0) % kRowsPerBlock == 0) {
      endRow = false;
    }
  }

  if (endCol) {
    for (int col = colStart + threadIdx.x;
         col < input.getSize(0); col += blockDim.x) {
      T val = input[col];

      if (endRow) {
        for (int row = rowStart; row < output.getSize(0); ++row) {
          T out = output[row][col].ldg();
          out = Math<T>::add(out, val);
          output[row][col] = out;
        }
      } else {
        T rows[kRowUnroll];

        for (int row = rowStart; row < rowEnd; row += kRowUnroll) {
#pragma unroll
          for (int i = 0; i < kRowUnroll; ++i) {
            rows[i] = output[row + i][col].ldg();
          }

#pragma unroll
          for (int i = 0; i < kRowUnroll; ++i) {
            rows[i] = Math<T>::add(rows[i], val);
          }

#pragma unroll
          for (int i = 0; i < kRowUnroll; ++i) {
            output[row + i][col] = rows[i];
          }
        }
      }
    }
  } else {
    int col = colStart + threadIdx.x;

    T val[kColLoad];

#pragma unroll
    for (int i = 0; i < kColLoad; ++i) {
      val[i] = input[col + i * blockDim.x];
    }

    if (endRow) {
      for (int row = rowStart; row < output.getSize(0); ++row) {
#pragma unroll
        for (int i = 0; i < kColLoad; ++i) {
          T out = output[row][col + i * blockDim.x].ldg();
          out = Math<T>::add(out, val[i]);
          output[row][col + i * blockDim.x] = out;
        }
      }
    } else {
      T rows[kRowUnroll * kColLoad];

      for (int row = rowStart; row < rowEnd; row += kRowUnroll) {
#pragma unroll
        for (int i = 0; i < kRowUnroll; ++i) {
#pragma unroll
          for (int j = 0; j < kColLoad; ++j) {
            rows[i * kColLoad + j] =
              output[row + i][col + j * blockDim.x].ldg();
          }
        }

#pragma unroll
        for (int i = 0; i < kRowUnroll; ++i) {
#pragma unroll
          for (int j = 0; j < kColLoad; ++j) {
            rows[i * kColLoad + j] =
              Math<T>::add(rows[i * kColLoad + j], val[j]);
          }
        }

#pragma unroll
        for (int i = 0; i < kRowUnroll; ++i) {
#pragma unroll
          for (int j = 0; j < kColLoad; ++j) {
            output[row + i][col + j * blockDim.x] =
              rows[i * kColLoad + j];
          }
        }
      }
    }
  }
}

template <typename T, int kRowsPerBlock, int kRowUnroll, int kColLoad>
__global__ void assignAlongColumns(Tensor<T, 1, true> input,
                                   Tensor<T, 2, true> output) {
  static_assert(kRowsPerBlock % kRowUnroll == 0, "must fit rows");

  // blockIdx.x: which chunk of rows we are responsible for updating
  // blockIdx.y: which chunk of columns we are responsible for
  // updating
  int rowStart = blockIdx.x * kRowsPerBlock;
  int rowEnd = rowStart + kRowsPerBlock;
  int colStart = blockIdx.y * blockDim.x * kColLoad;

  // FIXME: if we have exact multiples, don't need this
  bool endRow = (blockIdx.x == gridDim.x - 1);
  bool endCol = (blockIdx.y == gridDim.y - 1);

  if (endRow) {
    if (output.getSize(0) % kRowsPerBlock == 0) {
      endRow = false;
    }
  }

  if (endCol) {
    for (int col = colStart + threadIdx.x;
         col < input.getSize(0); col += blockDim.x) {
      T val = input[col];

      if (endRow) {
        for (int row = rowStart; row < output.getSize(0); ++row) {
          output[row][col] = val;
        }
      } else {
        for (int row = rowStart; row < rowEnd; row += kRowUnroll) {
#pragma unroll
          for (int i = 0; i < kRowUnroll; ++i) {
            output[row + i][col] = val;
          }
        }
      }
    }
  } else {
    int col = colStart + threadIdx.x;

    T val[kColLoad];

#pragma unroll
    for (int i = 0; i < kColLoad; ++i) {
      val[i] = input[col + i * blockDim.x];
    }

    if (endRow) {
      for (int row = rowStart; row < output.getSize(0); ++row) {
#pragma unroll
        for (int i = 0; i < kColLoad; ++i) {
          output[row][col + i * blockDim.x] = val[i];
        }
      }
    } else {
      for (int row = rowStart; row < rowEnd; row += kRowUnroll) {
#pragma unroll
        for (int i = 0; i < kRowUnroll; ++i) {
#pragma unroll
          for (int j = 0; j < kColLoad; ++j) {
            output[row + i][col + j * blockDim.x] = val[j];
          }
        }
      }
    }
  }
}


template <typename T>
__global__ void sumAlongRowsWithGraph(
                     Tensor<int, 2, true> outIndex,
                     Tensor<int, 2, true> graphIndices,
                      Tensor<T, 2, true>   productDistances,
                      Tensor<T, 2, true> output) {
  __shared__ T sval;

  int row = blockIdx.x;

 // if (threadIdx.x == 0) {
  //  sval = input[row];
 // }

  __syncthreads();

  //T val = sval;

  // FIXME: speed up
  for (int i = threadIdx.x; i < output.getSize(1); i += blockDim.x) {
    int indice = i/graphIndices.getSize(1);
    int pos = i%graphIndices.getSize(1);
    int c =outIndex[row][indice];
    int s = graphIndices[c][pos];
    T out = productDistances[row][s];
    T val1 = productDistances[row][c];
    out = Math<T>::sub(out, val1);
    output[row][i] = out;
  }
  __syncthreads();
}

template <typename T, int NumWarpQ, int NumThreadQ, int ThreadsPerBlock>
__global__ void sumAlongRowsWithOrder(
                    Tensor<float, 2, true> graphDistancesBuf,
                     Tensor<int, 2, true> outIndex,
                     Tensor<int, 2, true> graphIndices,
                     Tensor<float, 2, true> graphDists,
                      Tensor<T, 2, true>   productDistances,
                      Tensor<float, 2, true> outputGraph,
                      Tensor<float, 2, true> output,
                      Tensor<int, 2, true> outputIndices,
                      int k,int begin ,int end, float initK) {
      // Each block handles a single row of the distances (results)
      constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

      __shared__ float smemK[kNumWarps * NumWarpQ];
      __shared__ int smemV[kNumWarps * NumWarpQ];
      __shared__ float smemK1[NumWarpQ];
      __shared__ int smemV1[NumWarpQ];

      int num =outIndex.getSize(1)*graphIndices.getSize(1);
      int nedge = graphIndices.getSize(1);
     if(k<=NumWarpQ){
      BlockSelect<float, int, false, Comparator<float>,
                 NumWarpQ, NumThreadQ, ThreadsPerBlock>
       heap(initK, -1, smemK, smemV, NumWarpQ);

      int row = blockIdx.x;

     //  if (threadIdx.x == 0) {
    //   sval = queryNorms[row];
    //  }

     //  __syncthreads();

   // Whole warps must participate in the selection
      int limit = utils::roundDown(num, kWarpSize);
      uint i = threadIdx.x;

      for (; i < limit; i += blockDim.x) {
        int indice = i/nedge;
        int pos = i%nedge;
        int c =outIndex[row][indice];
        if(c>=begin&&c<=end){
                int s = graphIndices[c][pos];
        T a2 = productDistances[row][s];
        T b2 = productDistances[row][c];
        float c2 = graphDists[c][pos];
        float v = a2-b2;
        graphDistancesBuf[row][i]=v;
        v-= c2;
        float dist = b2 - 0.25f*v*v/c2;

        heap.add(dist, i);
        }else{

        heap.add(initK, i);
        }

      }

      if (i < num) {
        int indice = i/nedge;
        int pos = i%nedge;
        int c =outIndex[row][indice];
        if(c>=begin&&c<=end){
        int s = graphIndices[c][pos];
        T a2 =  productDistances[row][s];
        T b2 =   productDistances[row][c];
        float c2 = graphDists[c][pos];
        float v =a2-b2;
        graphDistancesBuf[row][i]=v;
        v-= c2;
        float dist = b2 - 0.25f*v*v/c2;
        //outputIndices[row][i]= c*nedge+pos;

        heap.addThreadQ(dist, i);
        }else{
            heap.addThreadQ(initK, i);
        }
      }

      heap.reduce();

      for (int i = threadIdx.x; i < k; i += blockDim.x) {

         float v =  graphDistancesBuf[row][smemV[i]];
         outputGraph[row][i]= v;
         int indice = smemV[i]/nedge;
          int pos = smemV[i]%nedge;
          int c =outIndex[row][indice];
           T b2= productDistances[row][c];
           output[row][i] =b2;
          outputIndices[row][i] = c*nedge+pos;

      }
    }else{

         BlockSelect<float, int, false, Comparator<float>,
                 NumWarpQ, NumThreadQ, ThreadsPerBlock>
       heap(initK, -1, smemK, smemV, NumWarpQ);

      int row = blockIdx.x;

     //  if (threadIdx.x == 0) {
     //   sval = queryNorms[row];
     // }

      // __syncthreads();

   // Whole warps must participate in the selection
      int limit = utils::roundDown(num, kWarpSize);
      uint i = threadIdx.x;

      for (; i < limit; i += blockDim.x) {
        int indice = i/nedge;
        int pos = i%nedge;
        int c =outIndex[row][indice];
       if(c>=begin&&c<=end){
                int s = graphIndices[c][pos];
        T a2 = productDistances[row][s];
        T b2 = productDistances[row][c];
        float c2 = graphDists[c][pos];
        float v = a2-b2;
        graphDistancesBuf[row][i]=v;
        v-= c2;
        float dist = b2 - 0.25f*v*v/c2;

        heap.add(dist, i);
        }else{

        heap.add(initK, i);
        }
      }

      if (i < num) {
        int indice = i/nedge;
        int pos = i%nedge;
        int c =outIndex[row][indice];
     if(c>=begin&&c<=end){
        int s = graphIndices[c][pos];
        T a2 =  productDistances[row][s];
        T b2 =   productDistances[row][c];
        float c2 = graphDists[c][pos];
        float v =a2-b2;
        graphDistancesBuf[row][i]=v;
        v-= c2;
        float dist = b2 - 0.25f*v*v/c2;
        //outputIndices[row][i]= c*nedge+pos;

        heap.addThreadQ(dist, i);
        }else{
            heap.addThreadQ(initK, i);
        }
      }

      heap.reduce();



      for (int i = threadIdx.x; i < NumWarpQ; i += blockDim.x) {

         float v =  graphDistancesBuf[row][smemV[i]];
         outputGraph[row][i]= v;
         int indice = smemV[i]/nedge;
          int pos = smemV[i]%nedge;
          int c =outIndex[row][indice];
           T b2= productDistances[row][c];
           output[row][i] =b2;
          outputIndices[row][i] = c*nedge+pos;

      }


      for (int j = threadIdx.x; j < NumWarpQ; j += blockDim.x) {

                smemK1[j] = smemK[NumWarpQ+j];
                smemV1[j] =smemV[NumWarpQ+j];
       }




       BlockSelect<float, int, false, Comparator<float>,
                 NumWarpQ, NumThreadQ, ThreadsPerBlock>
       heap1(initK, -1, smemK, smemV, NumWarpQ);
       limit = utils::roundDown(NumWarpQ, kWarpSize);
       i = threadIdx.x;

      for (; i < limit; i += blockDim.x) {
         heap1.add(smemK1[i], smemV1[i]);
      }

      if (i < NumWarpQ) {
        heap1.addThreadQ(smemK1[i], smemV1[i]);
      }
       heap1.reduce();

      for (int i = threadIdx.x; i < k-NumWarpQ; i += blockDim.x) {
         float v =  graphDistancesBuf[row][smemV[i]];
         outputGraph[row][NumWarpQ+i]= v;
         int indice = smemV[i]/nedge;
          int pos = smemV[i]%nedge;
          int c =outIndex[row][indice];
           T b2= productDistances[row][c];
           output[row][NumWarpQ+i] =b2;
          outputIndices[row][NumWarpQ+i] = c*nedge+pos;

      }

    }


}



template <typename T, int NumWarpQ, int NumThreadQ, int ThreadsPerBlock>
__global__ void sumAlongRowsWithOrder2(
                    Tensor<float, 2, true> graphDistancesBuf,
                     Tensor<int, 2, true> outIndex,
                     Tensor<int, 2, true> graphIndices,
                     Tensor<float, 2, true> graphDists,
                      Tensor<T, 2, true>   productDistances,
                      Tensor<float, 2, true> outputGraph,
                      Tensor<float, 2, true> output,
                      Tensor<int, 2, true> outputIndices,
                      int k,int begin ,int end, float initK) {
      // Each block handles a single row of the distances (results)
      constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;
      __shared__ float smemK[kNumWarps * NumWarpQ];
      __shared__ int smemV[kNumWarps * NumWarpQ];

      int num =outIndex.getSize(1)*graphIndices.getSize(1);
      int nedge = graphIndices.getSize(1);
         BlockSelect<float, int, false, Comparator<float>,
                 NumWarpQ, NumThreadQ, ThreadsPerBlock>
       heap(initK, -1, smemK, smemV, NumWarpQ);

      int row = blockIdx.x;

   // Whole warps must participate in the selection
      int limit = utils::roundDown(num, kWarpSize);
      uint i = threadIdx.x;

      for (; i < limit; i += blockDim.x) {
        int indice = i/nedge;
        int pos = i%nedge;
        int c =outIndex[row][indice];
        int s = graphIndices[c][pos];
        T a2 =  productDistances[row][s];
        T b2 =   productDistances[row][c];
        float c2 = graphDists[c][pos];
        float v =a2-b2;
        graphDistancesBuf[row][i]=v;
        v-= c2;

       float dist =(v>0)?b2:(b2 - 0.25f*v*v/c2);
       //float dist =(b2 - 0.25f*v*v/c2);
        heap.add(dist, i);

      }

      if (i < num) {
        int indice = i/nedge;
        int pos = i%nedge;
        int c =outIndex[row][indice];
        int s = graphIndices[c][pos];
        T a2 =  productDistances[row][s];
        T b2 =   productDistances[row][c];
        float c2 = graphDists[c][pos];
        float v =a2-b2;
        graphDistancesBuf[row][i]=v;
        v-= c2;
         float dist =(v>0)?b2:(b2 - 0.25f*v*v/c2);
       //float dist =(b2 - 0.25f*v*v/c2);
        heap.addThreadQ(dist, i);
      }

      heap.reduce();


      for(int begin =0; begin<k;begin+=NumWarpQ){
         int k1 = min(NumWarpQ, k-begin);
        for (int i = threadIdx.x; i < k1; i += blockDim.x) {
         float v =  graphDistancesBuf[row][smemV[i]];
         outputGraph[row][begin+i]= v;
         int indice = smemV[i]/nedge;
          int pos = smemV[i]%nedge;
          int c =outIndex[row][indice];
           T b2= productDistances[row][c];
           output[row][begin+i] =b2;
          outputIndices[row][begin+i] = c*nedge+pos;
          smemK[i] = initK;

        }
        heap.reduce();
      }


}









template <typename T, int NumWarpQ, int NumThreadQ, int ThreadsPerBlock>
__global__ void sumAlongRowsWithOrder1(
                    Tensor<float, 2, true> graphDistancesBuf,
                     Tensor<int, 2, true> outIndex,
                     Tensor<int, 2, true> graphIndices,
                     Tensor<float, 2, true> graphDists,
                      Tensor<T, 2, true>   productDistances,
                      Tensor<float, 2, true> outputGraph,
                      Tensor<float, 2, true> output,
                      Tensor<int, 2, true> outputIndices,
                      int k, float initK) {
      // Each block handles a single row of the distances (results)
        extern __shared__ char smem[];

      //__shared__ T sval;
      int num =outIndex.getSize(1)*graphIndices.getSize(1);
      int nedge = graphIndices.getSize(1);

       float * smemK = (float *) smem;
       float * smemIter = (float *) smem;
       smemIter+=num;
       uint* smemV = (uint *)smemIter;



      int row = blockIdx.x;

       //__syncthreads();

        int limit = utils::roundDown(num, kWarpSize);
      uint i = threadIdx.x;

      for (; i < limit; i += blockDim.x) {
        int indice = i/nedge;
        int pos = i%nedge;
        int c =outIndex[row][indice];
        int s = graphIndices[c][pos];
        T a2 = productDistances[row][s];
        T b2 = productDistances[row][c];
        float c2 = graphDists[c][pos];
        float v = a2-b2;
        graphDistancesBuf[row][i]=v;
        v-= c2;
        float dist = b2 - 0.25f*v*v/c2;
        smemK[i]=dist;
        smemV[i]=i;
      }

      if (i < num) {
        int indice = i/nedge;
        int pos = i%nedge;
        int c =outIndex[row][indice];
        int s = graphIndices[c][pos];
        T a2 =  productDistances[row][s];
        T b2 =   productDistances[row][c];
        float c2 = graphDists[c][pos];
        float v =a2-b2;
        graphDistancesBuf[row][i]=v;
        v-= c2;
        float dist = b2 - 0.25f*v*v/c2;
        smemK[i]=dist;
        smemV[i]=i;
      }





     // uint i = threadIdx.x;

      //for (; i < num; i += blockDim.x) {
     //   int indice = i/nedge;
     //   int pos = i%nedge;
     //   int c =outIndex[row][indice];
     //   int s = graphIndices[c][pos];
     //   T a2 = productDistances[row][s];
     //   T b2 = productDistances[row][c];
      //  float c2 = graphDists[c][pos];
      //  float v = a2-b2;
      //  graphDistancesBuf[row][i]=v;
      //  v-= c2;
      //  float dist = b2 - 0.25f*v*v/c2;
      //  smemK[i]=dist;
      //  smemV[i]=i;
      //}

      //heap.reduce();
        __syncthreads();

        bitonic3(smemK, smemV, num);

        // __syncthreads();
      for (int i = threadIdx.x; i < k; i += blockDim.x) {

         float v =  graphDistancesBuf[row][smemV[i]];
         outputGraph[row][i]= v;
         int indice = smemV[i]/nedge;
          int pos = smemV[i]%nedge;
          int c =outIndex[row][indice];
           T b2= productDistances[row][c];
           output[row][i] =b2;
          outputIndices[row][i] = c*nedge+pos;

      }

}

template <typename T, typename TVec>
__global__ void sumAlongRows(Tensor<T, 1, true> input,
                             Tensor<TVec, 2, true> output) {
  __shared__ T sval;

  int row = blockIdx.x;

  if (threadIdx.x == 0) {
    sval = input[row];
  }

  __syncthreads();

  T val = sval;

  // FIXME: speed up
  for (int i = threadIdx.x; i < output.getSize(1); i += blockDim.x) {
    TVec out = output[row][i];
    out = Math<TVec>::add(out, val);
    output[row][i] = out;
  }
}

template <typename T, typename TVec>
void runSumAlongColumns(Tensor<T, 1, true>& input,
                        Tensor<T, 2, true>& output,
                        cudaStream_t stream) {
  FAISS_ASSERT(input.getSize(0) == output.getSize(1));

  int threadsPerBlock = 256;
  constexpr int kRowUnroll = 4;
  constexpr int kRowsPerBlock = kRowUnroll * 4;
  constexpr int kColLoad = 4;

  auto block = dim3(threadsPerBlock);

  if (input.template canCastResize<TVec>() &&
      output.template canCastResize<TVec>()) {
    auto inputV = input.template castResize<TVec>();
    auto outputV = output.template castResize<TVec>();

    auto grid =
      dim3(utils::divUp(outputV.getSize(0), kRowsPerBlock),
           utils::divUp(outputV.getSize(1), threadsPerBlock * kColLoad));

    sumAlongColumns<TVec, kRowsPerBlock, kRowUnroll, kColLoad>
      <<<grid, block, 0, stream>>>(inputV, outputV);
  } else {
    auto grid =
      dim3(utils::divUp(output.getSize(0), kRowsPerBlock),
           utils::divUp(output.getSize(1), threadsPerBlock * kColLoad));

    sumAlongColumns<T, kRowsPerBlock, kRowUnroll, kColLoad>
      <<<grid, block, 0, stream>>>(input, output);
  }

  CUDA_TEST_ERROR();
}

void runSumAlongColumns(Tensor<float, 1, true>& input,
                        Tensor<float, 2, true>& output,
                        cudaStream_t stream) {

  runSumAlongColumns<float, float4>(input, output, stream);
}

#ifdef FAISS_USE_FLOAT16
void runSumAlongColumns(Tensor<half, 1, true>& input,
                        Tensor<half, 2, true>& output,
                        cudaStream_t stream) {
  runSumAlongColumns<half, half2>(input, output, stream);
}
#endif

template <typename T, typename TVec>
void runAssignAlongColumns(Tensor<T, 1, true>& input,
                           Tensor<T, 2, true>& output,
                           cudaStream_t stream) {
  FAISS_ASSERT(input.getSize(0) == output.getSize(1));

  int threadsPerBlock = 256;
  constexpr int kRowUnroll = 4;
  constexpr int kRowsPerBlock = kRowUnroll * 4;
  constexpr int kColLoad = 4;

  auto block = dim3(threadsPerBlock);

  if (input.template canCastResize<TVec>() &&
      output.template canCastResize<TVec>()) {
    auto inputV = input.template castResize<TVec>();
    auto outputV = output.template castResize<TVec>();

    auto grid =
      dim3(utils::divUp(outputV.getSize(0), kRowsPerBlock),
           utils::divUp(outputV.getSize(1), threadsPerBlock * kColLoad));

    assignAlongColumns<TVec, kRowsPerBlock, kRowUnroll, kColLoad>
      <<<grid, block, 0, stream>>>(inputV, outputV);
  } else {
    auto grid =
      dim3(utils::divUp(output.getSize(0), kRowsPerBlock),
           utils::divUp(output.getSize(1), threadsPerBlock * kColLoad));

    assignAlongColumns<T, kRowsPerBlock, kRowUnroll, kColLoad>
      <<<grid, block, 0, stream>>>(input, output);
  }

  CUDA_TEST_ERROR();
}


void runAssignAlongColumns(Tensor<float, 1, true>& input,
                           Tensor<float, 2, true>& output,
                           cudaStream_t stream) {
  runAssignAlongColumns<float, float4>(input, output, stream);
}

#ifdef FAISS_USE_FLOAT16
void runAssignAlongColumns(Tensor<half, 1, true>& input,
                           Tensor<half, 2, true>& output,
                           cudaStream_t stream) {
  runAssignAlongColumns<half, half2>(input, output, stream);
}
#endif



template <typename T>
void runSumAlongRowsWithGraph(Tensor<int, 2, true>& outIndexView,
                     Tensor<int, 2, true>& graphIndices,
                      Tensor<T, 2, true>&   productDistances,
                      Tensor<T, 2, true>& outGraphDistances,
                     cudaStream_t stream) {

    int threadsPerBlock =
      std::min(outGraphDistances.getSize(1), getMaxThreadsCurrentDevice());
    auto grid = dim3(outGraphDistances.getSize(0));
    auto block = dim3(threadsPerBlock);
    sumAlongRowsWithGraph<T><<<grid, block, 64, stream>>>(outIndexView,graphIndices,productDistances,outGraphDistances);
     //outputVecT<T>("outGraphDistances11",outGraphDistances.data(),6,stream);
  CUDA_TEST_ERROR();
}
template <typename T>
void runL2SelectMinGraph(Tensor<float, 2, true>& graphDistancesBuf,Tensor<int, 2, true>& outIndexView,
                     Tensor<int, 2, true>& graphIndices,
                        Tensor<float, 2, true>& graphDists,
                      Tensor<T, 2, true>&   productDistances,
                       Tensor<float, 2, true>& outGraphDistances,
                     Tensor<float, 2, true>& outDistances2nd,
                      Tensor<int, 2, true>& outIndices2nd,
                      int k,int begin ,int end,
                     cudaStream_t stream) {
        constexpr int kThreadsPerBlock = 128;

    auto block = dim3(kThreadsPerBlock);
    auto grid = dim3(outDistances2nd.getSize(0));
    int num = outIndexView.getSize(1)*graphIndices.getSize(1);
   int semSize = 2*num*sizeof(float);
    #define RUN_L2_SELECT(NUM_WARP_Q, NUM_THREAD_Q)                         \
    do {                                                                \
      sumAlongRowsWithOrder2<T, NUM_WARP_Q, NUM_THREAD_Q, kThreadsPerBlock>       \
        <<<grid, block, 0, stream>>>(graphDistancesBuf,outIndexView, graphIndices, \
                                     graphDists, productDistances,outGraphDistances,outDistances2nd,\
                                    outIndices2nd,        \
                                     k,  begin ,end,Limits<float>::getMax());           \
    } while (0)

    if (k <= 32) {
      RUN_L2_SELECT(32, 2);
    } else if (k <= 64) {
      RUN_L2_SELECT(64, 3);
    } else if (k <= 128) {
      RUN_L2_SELECT(128, 3);
    } else if (k <= 256) {
      RUN_L2_SELECT(256, 4);
    } else if (k <= 512) {
      RUN_L2_SELECT(512, 8);
    } else if (k <= 1024) {
      RUN_L2_SELECT(1024, 8);
    } else {
      RUN_L2_SELECT(1024, 8);
    }
   CUDA_TEST_ERROR();
}

void runSumAlongRowsWithGraph( Tensor<int, 2, true>& outIndexView,
                     Tensor<int, 2, true>& graphIndices,
                      Tensor<float, 2, true>&   productDistances,
                      Tensor<float, 2, true>& outGraphDistances,
                     cudaStream_t stream){


    runSumAlongRowsWithGraph<float>(outIndexView,graphIndices,productDistances,outGraphDistances, stream);
                     }

template <typename T>
__global__ void sumAlongColumnsGraph(Tensor<T, 1, true> input,
                             Tensor<T, 2, true> output) {

  int row = blockIdx.x;


  __syncthreads();


  // FIXME: speed up
  for (int i = threadIdx.x; i < output.getSize(1); i += blockDim.x) {
    T out = output[row][i];
    T in = input[i];
    out = Math<T>::add(out,in);
    output[row][i] = out;
  }
}


template <typename T>
void runSumAlongColumnsGraph(Tensor<T, 1, true>& input,
                           Tensor<T, 2, true>& output,
                           cudaStream_t stream) {
  FAISS_ASSERT(input.getSize(0) == output.getSize(1));

  int threadsPerBlock = 256;
  auto block = dim3(threadsPerBlock);

    auto grid =
      dim3(output.getSize(0));

    sumAlongColumnsGraph<T>
      <<<grid, block, 0, stream>>>(input, output);

  CUDA_TEST_ERROR();
}

void runSumAlongColumnsGraph1(Tensor<float, 1, true>& input,
                           Tensor<float, 2, true>& output,
                           cudaStream_t stream) {
  runSumAlongColumnsGraph<float>(input, output, stream);
}


void runL2SelectMinGraph(Tensor<float, 2, true>& graphDistancesBuf, Tensor<int, 2, true>& outIndexView,
                     Tensor<int, 2, true>& graphIndices,
                     Tensor<float, 2, true>& graphDists,
                      Tensor<float, 2, true>&   productDistances,
                         Tensor<float, 2, true>& outGraphDistances,
                      Tensor<float, 2, true>& outDistances2nd,
                      Tensor<int, 2, true>& outIndices2nd,int k,int begin ,int end,
                     cudaStream_t stream){


    runL2SelectMinGraph<float>(graphDistancesBuf,outIndexView,graphIndices,
                                graphDists,productDistances,
                               outGraphDistances,outDistances2nd,outIndices2nd,k,begin,end,stream);

                     }

#ifdef FAISS_USE_FLOAT16
void runSumAlongRowsWithGraph(Tensor<int, 2, true>& outIndexView,
                     Tensor<int, 2, true>& graphIndices,
                      Tensor<half, 2, true>&   productDistances,
                      Tensor<half, 2, true>& outGraphDistances,
                     cudaStream_t stream) {
  runSumAlongRowsWithGraph<half>(outIndexView,graphIndices,productDistances,outGraphDistances, stream);
}
#endif

template <typename T, typename TVec>
void runSumAlongRows(Tensor<T, 1, true>& input,
                     Tensor<T, 2, true>& output,
                     cudaStream_t stream) {
  FAISS_ASSERT(input.getSize(0) == output.getSize(0));

  if (output.template canCastResize<TVec>()) {
    auto outputV = output.template castResize<TVec>();

    int threadsPerBlock =
      std::min(outputV.getSize(1), getMaxThreadsCurrentDevice());
    auto grid = dim3(outputV.getSize(0));
    auto block = dim3(threadsPerBlock);

    sumAlongRows<T, TVec><<<grid, block, 0, stream>>>(input, outputV);
  } else {
    int threadsPerBlock =
      std::min(output.getSize(1), getMaxThreadsCurrentDevice());
    auto grid = dim3(output.getSize(0));
    auto block = dim3(threadsPerBlock);

    sumAlongRows<T, T><<<grid, block, 0, stream>>>(input, output);
  }

  CUDA_TEST_ERROR();
}

void runSumAlongRows(Tensor<float, 1, true>& input,
                     Tensor<float, 2, true>& output,
                     cudaStream_t stream) {
  runSumAlongRows<float, float4>(input, output, stream);
}

#ifdef FAISS_USE_FLOAT16
void runSumAlongRows(Tensor<half, 1, true>& input,
                     Tensor<half, 2, true>& output,
                     cudaStream_t stream) {
  runSumAlongRows<half, half2>(input, output, stream);
}
#endif

} } // namespace
