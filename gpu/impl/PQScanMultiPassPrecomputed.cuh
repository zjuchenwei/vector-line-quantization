/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../GpuIndicesOptions.h"
#include "../utils/Tensor.cuh"
#include "../utils/DeviceTensor.cuh"
#include "../utils/DeviceVector.cuh"
#include "../utils/NoTypeTensor.cuh"
#include <thrust/device_vector.h>

namespace faiss { namespace gpu {

class GpuResources;

void runPQScanMultiPassPrecomputedGraph(Tensor<float, 2, true>& queries,
                                   Tensor<float, 2, true>& precompTerm1,
                                   Tensor<float, 1, true>& subQuantizerNorms,
                                   NoTypeTensor<3, true>& precompTerm2,
                                   NoTypeTensor<3, true>& precompTerm3,
                                   Tensor<int, 2, true>& topQueryToCentroid,
                                   DeviceTensor<float, 2, true>& graphDistances,
                                   Tensor<int, 2, true>& edgeInfo,
                                   Tensor<float, 2, true>& edgeDistInfo,
                                   Tensor<float, 1, true>& lambdaInfo,
                                    Tensor<float, 1, true>&  constInfo,
                                   bool useFloat16Lookup,
                                   int bytesPerCode,
                                   int numSubQuantizers,
                                   int numSubQuantizerCodes,
                                   thrust::device_vector<void*>& listCodes,
                                   thrust::device_vector<void*>& listIndices,
                                   thrust::device_vector<void*>& listLambdas,
                                    thrust::device_vector<void*>& listConsts,
                                   IndicesOptions indicesOptions,
                                   thrust::device_vector<int>& listLengths,
                                   int maxListLength,
                                   int k,
                                   Tensor<int, 2, true>& coarseIndices2nd,
                                   // output
                                   Tensor<float, 2, true>& outDistances,
                                   // output
                                   Tensor<long, 2, true>& outIndices,
                                   GpuResources* res) ;
void runPQScanMultiPassPrecomputedGraph(Tensor<float, 2, true>& queries,
                                   Tensor<float, 2, true>& precompTerm1,
                                  Tensor<float, 1, true>& subQuantizerNorms,
                                   NoTypeTensor<3, true>& precompTerm2,
                                   NoTypeTensor<3, true>& precompTerm3,
                                   Tensor<int, 2, true>& topQueryToCentroid,
                                   DeviceTensor<float, 2, true>& graphDistances,
                                   Tensor<int, 2, true>& edgeInfo,
                                   Tensor<float, 2, true>& edgeDistInfo,
                                   Tensor<float, 1, true>& lambdaInfo,
                                   bool useFloat16Lookup,
                                   int bytesPerCode,
                                   int numSubQuantizers,
                                   int numSubQuantizerCodes,
                                   thrust::device_vector<void*>& listCodes,
                                   thrust::device_vector<void*>& listIndices,
                                   thrust::device_vector<void*>& listLambdas,
                                   IndicesOptions indicesOptions,
                                   thrust::device_vector<int>& listLengths,
                                   int maxListLength,
                                   int k,
                                   // output
                                   Tensor<float, 2, true>& outDistances,
                                   // output
                                   Tensor<long, 2, true>& outIndices,
                                   GpuResources* res) ;

void runPQScanMultiPassPrecomputed(Tensor<float, 2, true>& queries,
                                   Tensor<float, 2, true>& precompTerm1,
                                   NoTypeTensor<3, true>& precompTerm2,
                                   NoTypeTensor<3, true>& precompTerm3,
                                   Tensor<int, 2, true>& topQueryToCentroid,
                                   bool useFloat16Lookup,
                                   int bytesPerCode,
                                   int numSubQuantizers,
                                   int numSubQuantizerCodes,
                                   thrust::device_vector<void*>& listCodes,
                                   thrust::device_vector<void*>& listIndices,
                                   IndicesOptions indicesOptions,
                                   thrust::device_vector<int>& listLengths,
                                   int maxListLength,
                                   int k,
                                   // output
                                   Tensor<float, 2, true>& outDistances,
                                   // output
                                   Tensor<long, 2, true>& outIndices,
                                   GpuResources* res);

} } // namespace
