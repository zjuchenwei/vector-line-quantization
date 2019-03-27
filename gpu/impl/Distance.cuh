/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../utils/DeviceTensor.cuh"
#include "../utils/Float16.cuh"

namespace faiss { namespace gpu {

class GpuResources;



/// Calculates brute-force L2 distance between `vectors` and
/// `queries`, returning the k closest results seen
void runL2Distance(GpuResources* resources,
                   Tensor<float, 2, true>& vectors,
                   Tensor<float, 2, true>* vectorsTransposed,
                   // can be optionally pre-computed; nullptr if we
                   // have to compute it upon the call
                   Tensor<float, 1, true>* vectorNorms,
                   Tensor<float, 2, true>& queries,
                   int k,
                   Tensor<float, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices,
                   // Do we care about `outDistances`? If not, we can
                   // take shortcuts.
                   bool ignoreOutDistances = false,
                   // Hint to use a different sized tile for
                   // multi-streaming the queries. If <= 0, we use the
                   // default
                   int tileSizeOverride = -1);

/// Calculates brute-force inner product distance between `vectors`
/// and `queries`, returning the k closest results seen
void runIPDistance(GpuResources* resources,
                   Tensor<float, 2, true>& vectors,
                   Tensor<float, 2, true>* vectorsTransposed,
                   Tensor<float, 2, true>& queries,
                   int k,
                   Tensor<float, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices,
                   // Hint to use a different sized tile for
                   // multi-streaming the queries. If <= 0, we use the
                   // default
                   int tileSizeOverride = -1);
 void runL2DistanceWithGraph(GpuResources* resources,
              Tensor<float, 2, true>& vectors,
              Tensor<float, 2, true>* vectorsTransposed,
              Tensor<float, 1, true>* vectorNorms,
              Tensor<float, 2, true>& queries,
              int k,
              int k1,
              Tensor<int, 2, true>& graphIndices,
               Tensor<float, 2, true>& graphDists,
              Tensor<float, 2, true>& outDistances,
              Tensor<float, 2, true>& outGraphDistances,
              Tensor<int, 2, true>& outIndices,
              Tensor<float, 2, true>& outDistances2nd,
              Tensor<int, 2, true>& outIndices2nd,
              int begin ,int end,
              bool ignoreOutDistances,
              int tileSizeOverride= -1);

#ifdef FAISS_USE_FLOAT16
void runIPDistance(GpuResources* resources,
                   Tensor<half, 2, true>& vectors,
                   Tensor<half, 2, true>* vectorsTransposed,
                   Tensor<half, 2, true>& queries,
                   int k,
                   Tensor<half, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices,
                   bool useHgemm,
                   int tileSizeOverride = -1);

void runL2Distance(GpuResources* resources,
                   Tensor<half, 2, true>& vectors,
                   Tensor<half, 2, true>* vectorsTransposed,
                   Tensor<half, 1, true>* vectorNorms,
                   Tensor<half, 2, true>& queries,
                   int k,
                   Tensor<half, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices,
                   bool useHgemm,
                   bool ignoreOutDistances = false,
                   int tileSizeOverride = -1);
void
runL2DistanceWithGraph(GpuResources* resources,
              Tensor<float, 2, true>& vectors,
              Tensor<float, 2, true>* vectorsTransposed,
              Tensor<float, 1, true>* vectorNorms,
              Tensor<float, 2, true>& queries,
              int k,
              Tensor<int, 2, true>& graphIndices,
              Tensor<float, 2, true>& outDistances,
              Tensor<float, 2, true>& outGraphDistances,
              Tensor<int, 2, true>& outIndices,
              bool ignoreOutDistances,
              int tileSizeOverride=-1);
void
runL2DistanceWithGraph(GpuResources* resources,
              Tensor<half, 2, true>& vectors,
              Tensor<half, 2, true>* vectorsTransposed,
              Tensor<half, 1, true>* vectorNorms,
              Tensor<half, 2, true>& queries,
              int k,
              Tensor<int, 2, true>& graphIndices,
              Tensor<half, 2, true>& outDistances,
              Tensor<half, 2, true>& outGraphDistances,
              Tensor<int, 2, true>& outIndices,
              bool useHgemm,
              bool ignoreOutDistances,
              int tileSizeOverride= -1);
#endif

void testL2DistanceWithGraph(GpuResources* resources,
                   Tensor<float, 2, true>& centroids,
                   Tensor<float, 2, true>* centroidsTransposed,
                   Tensor<float, 1, true>* centroidNorms,
                   Tensor<float, 2, true>& queries,
                   int k,
                   int start,
                   Tensor<int, 2, true>& graphIndices,
                   Tensor<float, 2, true>& outDistances,
                   Tensor<float, 2, true>& outGraphDistances,
                   Tensor<int, 2, true>& outIndices,
                   bool useHgemm,
                   bool ignoreOutDistances = false,
                   int tileSizeOverride = -1);

} } // namespace
