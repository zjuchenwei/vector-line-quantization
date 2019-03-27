/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "IVFUtils.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/DeviceVector.cuh"
#include "../utils/StaticUtils.h"
#include "../utils/Tensor.cuh"
#include "../utils/ThrustAllocator.cuh"
#include "../utils/helper.cuh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace faiss { namespace gpu {

// Calculates the total number of intermediate distances to consider
// for all queries
__global__ void
getResultLengths(Tensor<int, 2, true> topQueryToCentroid,
                 int* listLengths,
                 int totalSize,
                 Tensor<int, 2, true> length) {
  int linearThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearThreadId >= totalSize) {
    return;
  }

  int nprobe = topQueryToCentroid.getSize(1);
  int queryId = linearThreadId / nprobe;
  int listId = linearThreadId % nprobe;

  int centroidId = topQueryToCentroid[queryId][listId];

  // Safety guard in case NaNs in input cause no list ID to be generated
  length[queryId][listId] = (centroidId != -1) ? listLengths[centroidId] : 0;
}


// Calculates the total number of intermediate distances to consider
// for all queries
__global__ void
getResultLengthsGraph(Tensor<int, 2, true> topQueryToCentroid,
                 int* listLengths,
                 int nedge,
                 int totalSize,
                 Tensor<int, 2, true> length) {
  int linearThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearThreadId >= totalSize) {
    return;
  }

  int nprobe = topQueryToCentroid.getSize(1);
  int queryId = linearThreadId / nprobe;
  int nprobeId = linearThreadId % nprobe;

  int listId = topQueryToCentroid[queryId][nprobeId];

  // Safety guard in case NaNs in input cause no list ID to be generated
  for(int i = 0;i<nedge;i++)
  length[queryId][nprobeId*nedge+i] = (listId != -1) ? listLengths[listId*nedge+i] : 0;
}

// Calculates the total number of intermediate distances to consider
// for all queries
__global__ void
getResultLengthsGraph(Tensor<int, 2, true> coarseIndices2nd,
                 int* listLengths,
                 int totalSize,
                 Tensor<int, 2, true> length) {
  int linearThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearThreadId >= totalSize) {
    return;
  }

  int nprobe = coarseIndices2nd.getSize(1);
  int queryId = linearThreadId / nprobe;
  int nprobeId = linearThreadId % nprobe;
  int listId = coarseIndices2nd[queryId][nprobeId];

  length[queryId][nprobeId] = (listId != -1) ? ((listLengths[listId]>1024)?1024:listLengths[listId]) : 0;
  //length[queryId][nprobeId] = (listId != -1) ? listLengths[listId] : 0;
 // printf("length[%d][%d]:%d",queryId,nprobeId,length[queryId][nprobeId]);
}


void runCalcListOffsetsGraph(Tensor<int, 2, true>& topQueryToCentroid,
                         int       nedge,
                        thrust::device_vector<int>& listLengths,
                        Tensor<int, 2, true>& prefixSumOffsets,
                        Tensor<char, 1, true>& thrustMem,
                        cudaStream_t stream) {
  FAISS_ASSERT(topQueryToCentroid.getSize(0) == prefixSumOffsets.getSize(0));
  FAISS_ASSERT(topQueryToCentroid.getSize(1)*nedge == prefixSumOffsets.getSize(1));

  int totalSize = topQueryToCentroid.numElements();

  int numThreads = std::min(totalSize, getMaxThreadsCurrentDevice());
  int numBlocks = utils::divUp(totalSize, numThreads);

  auto grid = dim3(numBlocks);
  auto block = dim3(numThreads);

  getResultLengthsGraph<<<grid, block, 0, stream>>>(
    topQueryToCentroid,
    listLengths.data().get(),
     nedge,
    totalSize,
    prefixSumOffsets);
  CUDA_TEST_ERROR();

  // Prefix sum of the indices, so we know where the intermediate
  // results should be maintained
  // Thrust wants a place for its temporary allocations, so provide
  // one, so it won't call cudaMalloc/Free
  GpuResourcesThrustAllocator alloc(thrustMem.data(),
                                    thrustMem.getSizeInBytes());

  thrust::inclusive_scan(thrust::cuda::par(alloc).on(stream),
                         prefixSumOffsets.data(),
                         prefixSumOffsets.data() + totalSize*nedge,
                         prefixSumOffsets.data());
  CUDA_TEST_ERROR();
}


void runCalcListOffsetsGraph(Tensor<int, 2, true>& coarseIndices2nd,
                        thrust::device_vector<int>& listLengths,
                        Tensor<int, 2, true>& prefixSumOffsets,
                        Tensor<char, 1, true>& thrustMem,
                        cudaStream_t stream) {
  FAISS_ASSERT(coarseIndices2nd.getSize(0) == prefixSumOffsets.getSize(0));
  FAISS_ASSERT(coarseIndices2nd.getSize(1) == prefixSumOffsets.getSize(1));

  int totalSize = coarseIndices2nd.numElements();

  int numThreads = std::min(totalSize, getMaxThreadsCurrentDevice());
  int numBlocks = utils::divUp(totalSize, numThreads);

  auto grid = dim3(numBlocks);
  auto block = dim3(numThreads);
  //outputVecInt("listLengths", listLengths.data().get(),100,stream);
  getResultLengthsGraph<<<grid, block, 0, stream>>>(
     coarseIndices2nd,
    listLengths.data().get(),
    totalSize,
    prefixSumOffsets);
// outputVecInt("prefixSumOffsets",prefixSumOffsets.data(),100,stream);
  CUDA_TEST_ERROR();

  // Prefix sum of the indices, so we know where the intermediate
  // results should be maintained
  // Thrust wants a place for its temporary allocations, so provide
  // one, so it won't call cudaMalloc/Free
  GpuResourcesThrustAllocator alloc(thrustMem.data(),
                                    thrustMem.getSizeInBytes());

thrust::inclusive_scan(thrust::cuda::par(alloc).on(stream),
                         prefixSumOffsets.data(),
                        prefixSumOffsets.data() + totalSize,
                        prefixSumOffsets.data());
  CUDA_TEST_ERROR();
}

void runCalcListOffsets(Tensor<int, 2, true>& topQueryToCentroid,
                        thrust::device_vector<int>& listLengths,
                        Tensor<int, 2, true>& prefixSumOffsets,
                        Tensor<char, 1, true>& thrustMem,
                        cudaStream_t stream) {
  FAISS_ASSERT(topQueryToCentroid.getSize(0) == prefixSumOffsets.getSize(0));
  FAISS_ASSERT(topQueryToCentroid.getSize(1) == prefixSumOffsets.getSize(1));

  int totalSize = topQueryToCentroid.numElements();

  int numThreads = std::min(totalSize, getMaxThreadsCurrentDevice());
  int numBlocks = utils::divUp(totalSize, numThreads);

  auto grid = dim3(numBlocks);
  auto block = dim3(numThreads);

  getResultLengths<<<grid, block, 0, stream>>>(
    topQueryToCentroid,
    listLengths.data().get(),
    totalSize,
    prefixSumOffsets);
  CUDA_TEST_ERROR();

  // Prefix sum of the indices, so we know where the intermediate
  // results should be maintained
  // Thrust wants a place for its temporary allocations, so provide
  // one, so it won't call cudaMalloc/Free
  GpuResourcesThrustAllocator alloc(thrustMem.data(),
                                    thrustMem.getSizeInBytes());

  thrust::inclusive_scan(thrust::cuda::par(alloc).on(stream),
                         prefixSumOffsets.data(),
                         prefixSumOffsets.data() + totalSize,
                         prefixSumOffsets.data());
  CUDA_TEST_ERROR();
}

} } // namespace
