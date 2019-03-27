/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../utils/Float16.cuh"
#include "../utils/Tensor.cuh"

namespace faiss { namespace gpu {

// output[x][i] += input[i] for all x
void runSumAlongColumns(Tensor<float, 1, true>& input,
                        Tensor<float, 2, true>& output,
                        cudaStream_t stream);

#ifdef FAISS_USE_FLOAT16
void runSumAlongColumns(Tensor<half, 1, true>& input,
                        Tensor<half, 2, true>& output,
                        cudaStream_t stream);
#endif

// output[x][i] = input[i] for all x
void runAssignAlongColumns(Tensor<float, 1, true>& input,
                           Tensor<float, 2, true>& output,
                           cudaStream_t stream);

#ifdef FAISS_USE_FLOAT16
void runAssignAlongColumns(Tensor<half, 1, true>& input,
                           Tensor<half, 2, true>& output,
                           cudaStream_t stream);
#endif

// output[i][x] += input[i] for all x
void runSumAlongRows(Tensor<float, 1, true>& input,
                     Tensor<float, 2, true>& output,
                     cudaStream_t stream);
void runSumAlongRowsWithGraph(Tensor<int, 2, true>& outIndexView,
                     Tensor<int, 2, true>& graphIndices,
                      Tensor<float, 2, true>&   productDistances,
                      Tensor<float, 2, true>& outGraphDistances,
                     cudaStream_t stream);
void runSumAlongColumnsGraph1(Tensor<float, 1, true>& input,
                           Tensor<float, 2, true>& output,
                           cudaStream_t stream);
void runL2SelectMinGraph(Tensor<float, 2, true>& graphDistancesBuf, Tensor<int, 2, true>& outIndexView,
                     Tensor<int, 2, true>& graphIndices,
                     Tensor<float, 2, true>& graphDists,
                      Tensor<float, 2, true>&  productDistances,
                      Tensor<float, 2, true>& outGraphDistances,
                      Tensor<float, 2, true>& outDistances2nd,
                      Tensor<int, 2, true>& outIndices2nd,int k,int begin ,int end,
                     cudaStream_t stream);
#ifdef FAISS_USE_FLOAT16
void runSumAlongRows(Tensor<half, 1, true>& input,
                     Tensor<half, 2, true>& output,
                     cudaStream_t stream);
void runSumAlongRowsWithGraph(Tensor<int, 2, true>& outIndexView,
                     Tensor<int, 2, true>& graphIndices,
                      Tensor<half, 2, true>&   productDistances,
                      Tensor<half, 2, true>& outGraphDistances,
                     cudaStream_t stream);
#endif

} } // namespace
