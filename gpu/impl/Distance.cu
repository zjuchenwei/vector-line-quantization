/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "Distance.cuh"
#include "BroadcastSum.cuh"
#include "L2Norm.cuh"
#include "L2Select.cuh"
#include "../../FaissAssert.h"
#include "../GpuResources.h"
#include "../utils/DeviceUtils.h"
#include "../utils/helper.cuh"
#include "../utils/Limits.cuh"
#include "../utils/MatrixMult.cuh"
#include "../utils/BlockSelectKernel.cuh"

#include <memory>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <iostream>
namespace faiss { namespace gpu {

namespace {

constexpr int kDefaultTileSize = 256;

template <typename T>
int chooseTileSize(int tileSizeOverride,
                   size_t numCentroids,
                   size_t tempMemAvailable) {
  if (tileSizeOverride > 0) {
    return tileSizeOverride;
  }

  size_t tileSize =
    sizeof(T) < 4 ? kDefaultTileSize * 2 : kDefaultTileSize;

  while (tileSize > 64) {
    size_t memRequirement = 2 * tileSize * numCentroids * sizeof(T);

    if (memRequirement <= tempMemAvailable) {
      // This fits entirely into our temporary memory
      return tileSize;
    }

    // Otherwise, halve the tile size
    tileSize /= 2;
  }

  // We use 64 as the minimum acceptable tile size
  FAISS_ASSERT(tileSize >= 64);

  // FIXME: if we're running with no available temp memory, do we try
  // and go larger based on free memory available on the device?

  return tileSize;
}

}

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

template <typename T>
void runL2DistanceWithGraph(GpuResources* resources,
                   Tensor<T, 2, true>& centroids,
                   Tensor<T, 2, true>* centroidsTransposed,
                   Tensor<T, 1, true>* centroidNorms,
                   Tensor<T, 2, true>& queries,
                   int k,
                   Tensor<int, 2, true>& graphIndices,
                   Tensor<T, 2, true>& outDistances,
                   Tensor<T, 2, true>& outGraphDistances,
                   Tensor<int, 2, true>& outIndices,

                   bool useHgemm,
                   bool ignoreOutDistances = false,
                   int tileSizeOverride = -1) {
  FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outDistances.getSize(1) == k);
  FAISS_ASSERT(outIndices.getSize(1) == k);

  auto& mem = resources->getMemoryManagerCurrentDevice();
  auto defaultStream = resources->getDefaultStreamCurrentDevice();

  // If we're quering against a 0 sized set, just return empty results
  if (centroids.numElements() == 0) {
    thrust::fill(thrust::cuda::par.on(defaultStream),
                 outDistances.data(), outDistances.end(),
                 Limits<T>::getMax());

    thrust::fill(thrust::cuda::par.on(defaultStream),
                 outIndices.data(), outIndices.end(),
                 -1);

    return;
  }
  //outputVecT<T>("centroidNorms",centroidNorms->data(),10,defaultStream);
  // If ||c||^2 is not pre-computed, calculate it
  DeviceTensor<T, 1, true> cNorms;
  if (!centroidNorms) {
     printf("centroidNorms!!\n");
    cNorms = std::move(DeviceTensor<T, 1, true>(
                       mem,
                       {centroids.getSize(0)}, defaultStream));
    runL2Norm(centroids, cNorms, true, defaultStream);
    centroidNorms = &cNorms;
  }
  //
  // Prepare norm vector ||q||^2; ||c||^2 is already pre-computed
  //
  int qNormSize[1] = {queries.getSize(0)};
  DeviceTensor<T, 1, true> queryNorms(mem, qNormSize, defaultStream);
  //outputVecT<T>("queries",queries.data(),10,defaultStream);
  // ||q||^2
  runL2Norm(queries, queryNorms, true, defaultStream);
   //outputVecT<T>("queryNorms",queryNorms.data(),10,defaultStream);
  //
  // Handle the problem in row tiles, to avoid excessive temporary
  // memory requests
  //

  FAISS_ASSERT(k <= centroids.getSize(0));
  FAISS_ASSERT(k <= 1024); // select limitation

  int tileSize =
    chooseTileSize<T>(
      tileSizeOverride,
      centroids.getSize(0),
      resources->getMemoryManagerCurrentDevice().getSizeAvailable());

  int maxQueriesPerIteration = std::min(tileSize, queries.getSize(0));

  // Temporary output memory space we'll use
  DeviceTensor<T, 2, true> distanceBuf1(
    mem, {maxQueriesPerIteration, centroids.getSize(0)}, defaultStream);
  DeviceTensor<T, 2, true> distanceBuf2(
    mem, {maxQueriesPerIteration, centroids.getSize(0)}, defaultStream);
  DeviceTensor<T, 2, true>* distanceBufs[2] =
    {&distanceBuf1, &distanceBuf2};

  auto streams = resources->getAlternateStreamsCurrentDevice();
  streamWait(streams, {defaultStream});

  int curStream = 0;

  for (int i = 0; i < queries.getSize(0); i += maxQueriesPerIteration) {
    int numQueriesForIteration = std::min(maxQueriesPerIteration,
                                          queries.getSize(0) - i);

    auto distanceBufView =
      distanceBufs[curStream]->narrowOutermost(0, numQueriesForIteration);
    auto queryView =
      queries.narrowOutermost(i, numQueriesForIteration);
    auto outDistanceView =
      outDistances.narrowOutermost(i, numQueriesForIteration);
     auto outGraphDistancesView =
      outGraphDistances.narrowOutermost(i, numQueriesForIteration);
    auto outIndexView =
      outIndices.narrowOutermost(i, numQueriesForIteration);
    auto queryNormNiew =
      queryNorms.narrowOutermost(i, numQueriesForIteration);

    // L2 distance is ||c||^2 - 2qc + ||q||^2

    // -2qc
    // (query id x dim) x (centroid id, dim)' = (query id, centroid id)
    runMatrixMult(distanceBufView, false,
                  queryView, false,
                  centroidsTransposed ? *centroidsTransposed : centroids,
                  centroidsTransposed ? false : true,
                  -2.0f, 0.0f, useHgemm,
                  resources->getBlasHandleCurrentDevice(),
                  streams[curStream]);

    // For L2 distance, we use this fused kernel that performs both
    // adding ||c||^2 to -2qc and k-selection, so we only need two
    // passes (one write by the gemm, one read here) over the huge
    // region of output memory
    runL2SelectMin(distanceBufView,
                   *centroidNorms,
                   outDistanceView,
                   outIndexView,
                   k,
                   streams[curStream]);

    if (!ignoreOutDistances) {
      //printf("top-k ||c||^2 - 2qc + ||q||^2 ");
      // expand (query id) to (query id, k) by duplicating along rows
      // top-k ||c||^2 - 2qc + ||q||^2 in the form (query id, k)
      runSumAlongRows(queryNormNiew, outDistanceView, streams[curStream]);
      runSumAlongRowsWithGraph(outIndexView,graphIndices,distanceBufView,outGraphDistancesView,streams[curStream]);
      //outputVecT<T>("outGraphDistancesView",outGraphDistancesView.data(),10,streams[curStream]);
    }

    curStream = (curStream + 1) % 2;
  }

  // Have the desired ordering stream wait on the multi-stream
  streamWait({defaultStream}, streams);
}

template <typename T>
void runL2DistanceWithGraph(GpuResources* resources,
                   Tensor<T, 2, true>& centroids,
                   Tensor<T, 2, true>* centroidsTransposed,
                   Tensor<T, 1, true>* centroidNorms,
                   Tensor<T, 2, true>& queries,
                   int k,
                   int k1,
                   Tensor<int, 2, true>& graphIndices,
                   Tensor<float, 2, true>& graphDists,
                   Tensor<T, 2, true>& outDistances,
                   Tensor<T, 2, true>& outGraphDistances,
                   Tensor<int, 2, true>& outIndices,
                    Tensor<T, 2, true>& outDistances2nd,
                   Tensor<int, 2, true>& outIndices2nd,
                    int begin ,int end,
                   bool useHgemm,
                   bool ignoreOutDistances = false,
                   int tileSizeOverride = -1) {
  FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outDistances.getSize(1) == k);
  FAISS_ASSERT(outIndices.getSize(1) == k);

  auto& mem = resources->getMemoryManagerCurrentDevice();
  auto defaultStream = resources->getDefaultStreamCurrentDevice();

  // If we're quering against a 0 sized set, just return empty results
  if (centroids.numElements() == 0) {
    thrust::fill(thrust::cuda::par.on(defaultStream),
                 outDistances.data(), outDistances.end(),
                 Limits<T>::getMax());

    thrust::fill(thrust::cuda::par.on(defaultStream),
                 outIndices.data(), outIndices.end(),
                 -1);

    return;
  }
  //outputVecT<T>("centroidNorms",centroidNorms->data(),10,defaultStream);
  // If ||c||^2 is not pre-computed, calculate it
  DeviceTensor<T, 1, true> cNorms;
  if (!centroidNorms) {
     printf("centroidNorms!!\n");
    cNorms = std::move(DeviceTensor<T, 1, true>(
                       mem,
                       {centroids.getSize(0)}, defaultStream));
    runL2Norm(centroids, cNorms, true, defaultStream);
    centroidNorms = &cNorms;
  }
  //
  // Prepare norm vector ||q||^2; ||c||^2 is already pre-computed
  //
  int qNormSize[1] = {queries.getSize(0)};
  //DeviceTensor<T, 1, true> queryNorms(mem, qNormSize, defaultStream);
  //outputVecT<T>("queries",queries.data(),10,defaultStream);
  // ||q||^2
 // runL2Norm(queries, queryNorms, true, defaultStream);
   //outputVecT<T>("queryNorms",queryNorms.data(),10,defaultStream);
  //
  // Handle the problem in row tiles, to avoid excessive temporary
  // memory requests
  //

  FAISS_ASSERT(k <= centroids.getSize(0));
  FAISS_ASSERT(k <= 1024); // select limitation

  int tileSize =
    chooseTileSize<T>(
      tileSizeOverride,
      centroids.getSize(0),
      resources->getMemoryManagerCurrentDevice().getSizeAvailable());

  int maxQueriesPerIteration = std::min(tileSize, queries.getSize(0));

  // Temporary output memory space we'll use
  DeviceTensor<T, 2, true> distanceBuf1(
    mem, {maxQueriesPerIteration, centroids.getSize(0)}, defaultStream);
  DeviceTensor<T, 2, true> distanceBuf2(
    mem, {maxQueriesPerIteration, centroids.getSize(0)}, defaultStream);
  DeviceTensor<T, 2, true>* distanceBufs[2] =
    {&distanceBuf1, &distanceBuf2};

// Temporary output memory space we'll use
  DeviceTensor<T, 2, true> graphDistancesBuf1(
    mem, {maxQueriesPerIteration, k*graphIndices.getSize(1)}, defaultStream);
  DeviceTensor<T, 2, true> graphDistancesBuf2(
    mem, {maxQueriesPerIteration, k*graphIndices.getSize(1)}, defaultStream);
  DeviceTensor<T, 2, true>* graphDistancesBufs[2] =
    {&graphDistancesBuf1, &graphDistancesBuf2};

  auto streams = resources->getAlternateStreamsCurrentDevice();
  streamWait(streams, {defaultStream});

  int curStream = 0;

  for (int i = 0; i < queries.getSize(0); i += maxQueriesPerIteration) {
    int numQueriesForIteration = std::min(maxQueriesPerIteration,
                                          queries.getSize(0) - i);

    auto distanceBufView =
      distanceBufs[curStream]->narrowOutermost(0, numQueriesForIteration);
    auto graphDistancesBufView =
      graphDistancesBufs[curStream]->narrowOutermost(0, numQueriesForIteration);
    auto queryView =
      queries.narrowOutermost(i, numQueriesForIteration);
    auto outDistanceView =
      outDistances.narrowOutermost(i, numQueriesForIteration);
     auto outGraphDistancesView =
      outGraphDistances.narrowOutermost(i, numQueriesForIteration);
    auto outIndexView =
      outIndices.narrowOutermost(i, numQueriesForIteration);
     auto outDistanceView2nd =
      outDistances2nd.narrowOutermost(i, numQueriesForIteration);
    auto outIndexView2nd =
      outIndices2nd.narrowOutermost(i, numQueriesForIteration);
    //auto queryNormNiew =
     // queryNorms.narrowOutermost(i, numQueriesForIteration);

    // L2 distance is ||c||^2 - 2qc + ||q||^2

    // -2qc
    // (query id x dim) x (centroid id, dim)' = (query id, centroid id)
    runMatrixMult(distanceBufView, false,
                  queryView, false,
                  centroidsTransposed ? *centroidsTransposed : centroids,
                  centroidsTransposed ? false : true,
                  -2.0f, 0.0f, useHgemm,
                  resources->getBlasHandleCurrentDevice(),
                  streams[curStream]);

    // For L2 distance, we use this fused kernel that performs both
    // adding ||c||^2 to -2qc and k-selection, so we only need two
    // passes (one write by the gemm, one read here) over the huge
    // region of output memory
    runL2SelectMin(distanceBufView,
                   *centroidNorms,
                   outDistanceView,
                   outIndexView,
                   k,
                   streams[curStream]);
    runL2SelectMinGraph(graphDistancesBufView,outIndexView,graphIndices,graphDists,distanceBufView,
          outGraphDistancesView,outDistanceView2nd,outIndexView2nd,k1,begin ,end,streams[curStream]);


    curStream = (curStream + 1) % 2;
  }

  // Have the desired ordering stream wait on the multi-stream
  streamWait({defaultStream}, streams);
}



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
                   bool ignoreOutDistances,
                   int tileSizeOverride) {

  FAISS_ASSERT(outDistances.getSize(1) == k);
  FAISS_ASSERT(outIndices.getSize(1) == k);

  auto& mem = resources->getMemoryManagerCurrentDevice();
  auto defaultStream = resources->getDefaultStreamCurrentDevice();
  Tensor<float, 2, true> subcentroids(centroids.data()+start*centroids.getSize(1),
            {k, centroids.getSize(1)});
   Tensor<int, 2, true> subgraphIndices(graphIndices.data()+start*graphIndices.getSize(1),
            {k, graphIndices.getSize(1)});

  // If we're quering against a 0 sized set, just return empty results
  if (centroids.numElements() == 0) {
    thrust::fill(thrust::cuda::par.on(defaultStream),
                 outDistances.data(), outDistances.end(),
                 Limits<float>::getMax());

    thrust::fill(thrust::cuda::par.on(defaultStream),
                 outIndices.data(), outIndices.end(),
                 -1);

    return;
  }
  //outputVecT<T>("centroidNorms",centroidNorms->data(),10,defaultStream);
  // If ||c||^2 is not pre-computed, calculate it
  DeviceTensor<float, 1, true> cNorms;

    cNorms = std::move(DeviceTensor<float, 1, true>(
                       mem,
                       {k}, defaultStream));
    runL2Norm(subcentroids, cNorms, true, defaultStream);
    //centroidNorms = &cNorms;

  //
  // Prepare norm vector ||q||^2; ||c||^2 is already pre-computed
  //
  int qNormSize[1] = {queries.getSize(0)};
  DeviceTensor<float, 1, true> queryNorms(mem, qNormSize, defaultStream);
  //outputVecT<T>("queries",queries.data(),10,defaultStream);
  // ||q||^2
  runL2Norm(queries, queryNorms, true, defaultStream);
   //outputVecT<T>("queryNorms",queryNorms.data(),10,defaultStream);
  //
  // Handle the problem in row tiles, to avoid excessive temporary
  // memory requests
  //


  FAISS_ASSERT(k <= subcentroids.getSize(0));
  FAISS_ASSERT(k <= 1024); // select limitation

 // Temporary output memory space we'll use
  DeviceTensor<float, 2, true> subdistanceBuf1(
    mem, {10, subcentroids.getSize(0)}, defaultStream);
  DeviceTensor<float, 2, true> subdistanceBuf2(
    mem, {10, subcentroids.getSize(0)}, defaultStream);
  DeviceTensor<float, 2, true>* subdistanceBufs[2] =
    {&subdistanceBuf1, &subdistanceBuf2};

// Temporary output memory space we'll use
  DeviceTensor<float, 2, true> outDistanceBuf(
    mem, {10, k}, defaultStream);
  DeviceTensor<int, 2, true> outIndicesBuf(
    mem, {10, k}, defaultStream);



  // Temporary output memory space we'll use
  DeviceTensor<float, 2, true> distanceBuf1(
    mem, {10, centroids.getSize(0)}, defaultStream);
  DeviceTensor<float, 2, true> distanceBuf2(
    mem, {10, centroids.getSize(0)}, defaultStream);
  DeviceTensor<float, 2, true>* distanceBufs[2] =
    {&distanceBuf1, &distanceBuf2};

      // Temporary output memory space we'll use
  DeviceTensor<float, 2, true> distanceGraphBuf1(
    mem, {10, centroids.getSize(0)*32}, defaultStream);
  DeviceTensor<float, 2, true> distanceGraphBuf2(
    mem, {10, centroids.getSize(0)*32}, defaultStream);
  DeviceTensor<float, 2, true>* distanceGraphBufs[2] =
    {&distanceBuf1, &distanceBuf2};

  auto streams = resources->getAlternateStreamsCurrentDevice();
  streamWait(streams, {defaultStream});

  int curStream = 0;


    auto subdistanceBufView =
      subdistanceBufs[curStream]->narrowOutermost(0, 10);

    auto distanceBufView =
      distanceBufs[curStream]->narrowOutermost(0, 10);
        auto graphDistancesBufView =
      distanceGraphBufs[curStream]->narrowOutermost(0, 10);

    auto queryView =
      queries.narrowOutermost(0, 10);
    auto outDistanceView =
      outDistances.narrowOutermost(0, 10);
     auto outGraphDistancesView =
      outGraphDistances.narrowOutermost(0, 10);
    auto outIndexView =
      outIndices.narrowOutermost(0, 10);
    auto queryNormNiew =
      queryNorms.narrowOutermost(0, 10);

    auto outDistanceView2nd =
      outDistances.narrowOutermost(0, 10);
    auto outIndexView2nd =
      outIndices.narrowOutermost(0, 10);

    // L2 distance is ||c||^2 - 2qc + ||q||^2

    // -2qc
    // (query id x dim) x (centroid id, dim)' = (query id, centroid id)
    runMatrixMult(subdistanceBufView, false,
                  queryView, false,
                  subcentroids,
                  centroidsTransposed ? false : true,
                  -2.0f, 0.0f, useHgemm,
                  resources->getBlasHandleCurrentDevice(),
                  streams[curStream]);

    // For L2 distance, we use this fused kernel that performs both
    // adding ||c||^2 to -2qc and k-selection, so we only need two
    // passes (one write by the gemm, one read here) over the huge
    // region of output memory
    runL2SelectMin(subdistanceBufView,
                   cNorms,
                   outDistanceView,
                   outIndexView,
                   k,
                   streams[curStream]);

    if (!ignoreOutDistances) {
      //printf("top-k ||c||^2 - 2qc + ||q||^2 ");
      // expand (query id) to (query id, k) by duplicating along rows
      // top-k ||c||^2 - 2qc + ||q||^2 in the form (query id, k)
      runSumAlongRows(queryNormNiew, outDistanceView, streams[curStream]);
    }

    // -2qc
    // (query id x dim) x (centroid id, dim)' = (query id, centroid id)
    runMatrixMult(distanceBufView, false,
                  queryView, false,
                  centroids,
                  centroidsTransposed ? false : true,
                  -2.0f, 0.0f, useHgemm,
                  resources->getBlasHandleCurrentDevice(),
                  streams[curStream]);

      runL2SelectMin(distanceBufView,
                   *centroidNorms,
                   outDistanceBuf,
                   outIndicesBuf,
                   k,
                   streams[curStream]);

    if (!ignoreOutDistances) {
      //printf("top-k ||c||^2 - 2qc + ||q||^2 ");
      // expand (query id) to (query id, k) by duplicating along rows
      // top-k ||c||^2 - 2qc + ||q||^2 in the form (query id, k)
      //runSumAlongRowsWithGraph(outIndexView,subgraphIndices,distanceBufView,outGraphDistancesView,streams[curStream]);
    //runL2SelectMinGraph(graphDistancesBufView,outIndexView,graphIndices,graphDists,distanceBufView,
      //    outGraphDistancesView,outDistanceView2nd,outIndexView2nd,32,streams[curStream]);



    }
    curStream = (curStream + 1) % 2;

  // Have the desired ordering stream wait on the multi-stream
  streamWait({defaultStream}, streams);
}


template <typename T>
void runL2Distance(GpuResources* resources,
                   Tensor<T, 2, true>& centroids,
                   Tensor<T, 2, true>* centroidsTransposed,
                   Tensor<T, 1, true>* centroidNorms,
                   Tensor<T, 2, true>& queries,
                   int k,
                   Tensor<T, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices,
                   bool useHgemm,
                   bool ignoreOutDistances = false,
                   int tileSizeOverride = -1) {
  FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outDistances.getSize(1) == k);
  FAISS_ASSERT(outIndices.getSize(1) == k);

  auto& mem = resources->getMemoryManagerCurrentDevice();
  auto defaultStream = resources->getDefaultStreamCurrentDevice();

  // If we're quering against a 0 sized set, just return empty results
  if (centroids.numElements() == 0) {
    thrust::fill(thrust::cuda::par.on(defaultStream),
                 outDistances.data(), outDistances.end(),
                 Limits<T>::getMax());

    thrust::fill(thrust::cuda::par.on(defaultStream),
                 outIndices.data(), outIndices.end(),
                 -1);

    return;
  }

  // If ||c||^2 is not pre-computed, calculate it
  DeviceTensor<T, 1, true> cNorms;
  if (!centroidNorms) {
    cNorms = std::move(DeviceTensor<T, 1, true>(
                       mem,
                       {centroids.getSize(0)}, defaultStream));
    runL2Norm(centroids, cNorms, true, defaultStream);
    centroidNorms = &cNorms;
  }

  //
  // Prepare norm vector ||q||^2; ||c||^2 is already pre-computed
  //
  int qNormSize[1] = {queries.getSize(0)};
  DeviceTensor<T, 1, true> queryNorms(mem, qNormSize, defaultStream);

  // ||q||^2
  runL2Norm(queries, queryNorms, true, defaultStream);

  //
  // Handle the problem in row tiles, to avoid excessive temporary
  // memory requests
  //

  FAISS_ASSERT(k <= centroids.getSize(0));
  FAISS_ASSERT(k <= 1024); // select limitation

  int tileSize =
    chooseTileSize<T>(
      tileSizeOverride,
      centroids.getSize(0),
      resources->getMemoryManagerCurrentDevice().getSizeAvailable());

  int maxQueriesPerIteration = std::min(tileSize, queries.getSize(0));

  // Temporary output memory space we'll use
  DeviceTensor<T, 2, true> distanceBuf1(
    mem, {maxQueriesPerIteration, centroids.getSize(0)}, defaultStream);
  DeviceTensor<T, 2, true> distanceBuf2(
    mem, {maxQueriesPerIteration, centroids.getSize(0)}, defaultStream);
  DeviceTensor<T, 2, true>* distanceBufs[2] =
    {&distanceBuf1, &distanceBuf2};

  auto streams = resources->getAlternateStreamsCurrentDevice();
  streamWait(streams, {defaultStream});

  int curStream = 0;

  for (int i = 0; i < queries.getSize(0); i += maxQueriesPerIteration) {
    int numQueriesForIteration = std::min(maxQueriesPerIteration,
                                          queries.getSize(0) - i);

    auto distanceBufView =
      distanceBufs[curStream]->narrowOutermost(0, numQueriesForIteration);
    auto queryView =
      queries.narrowOutermost(i, numQueriesForIteration);
    auto outDistanceView =
      outDistances.narrowOutermost(i, numQueriesForIteration);
    auto outIndexView =
      outIndices.narrowOutermost(i, numQueriesForIteration);
    auto queryNormNiew =
      queryNorms.narrowOutermost(i, numQueriesForIteration);

    // L2 distance is ||c||^2 - 2qc + ||q||^2

    // -2qc
    // (query id x dim) x (centroid id, dim)' = (query id, centroid id)
    runMatrixMult(distanceBufView, false,
                  queryView, false,
                  centroidsTransposed ? *centroidsTransposed : centroids,
                  centroidsTransposed ? false : true,
                  -2.0f, 0.0f, useHgemm,
                  resources->getBlasHandleCurrentDevice(),
                  streams[curStream]);

    // For L2 distance, we use this fused kernel that performs both
    // adding ||c||^2 to -2qc and k-selection, so we only need two
    // passes (one write by the gemm, one read here) over the huge
    // region of output memory
    runL2SelectMin(distanceBufView,
                   *centroidNorms,
                   outDistanceView,
                   outIndexView,
                   k,
                   streams[curStream]);

    if (!ignoreOutDistances) {
      //printf("top-k ||c||^2 - 2qc + ||q||^2 ");
      // expand (query id) to (query id, k) by duplicating along rows
      // top-k ||c||^2 - 2qc + ||q||^2 in the form (query id, k)
      runSumAlongRows(queryNormNiew, outDistanceView, streams[curStream]);
    }

    curStream = (curStream + 1) % 2;
  }

  // Have the desired ordering stream wait on the multi-stream
  streamWait({defaultStream}, streams);
}

template <typename T>
void runIPDistance(GpuResources* resources,
                   Tensor<T, 2, true>& centroids,
                   Tensor<T, 2, true>* centroidsTransposed,
                   Tensor<T, 2, true>& queries,
                   int k,
                   Tensor<T, 2, true>& outDistances,
                   Tensor<int, 2, true>& outIndices,
                   bool useHgemm,
                   int tileSizeOverride = -1) {
  FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outDistances.getSize(1) == k);
  FAISS_ASSERT(outIndices.getSize(1) == k);

  auto& mem = resources->getMemoryManagerCurrentDevice();
  auto defaultStream = resources->getDefaultStreamCurrentDevice();

  // If we're quering against a 0 sized set, just return empty results
  if (centroids.numElements() == 0) {
    thrust::fill(thrust::cuda::par.on(defaultStream),
                 outDistances.data(), outDistances.end(),
                 Limits<T>::getMax());

    thrust::fill(thrust::cuda::par.on(defaultStream),
                 outIndices.data(), outIndices.end(),
                 -1);

    return;
  }

  //
  // Handle the problem in row tiles, to avoid excessive temporary
  // memory requests
  //

  FAISS_ASSERT(k <= centroids.getSize(0));
  FAISS_ASSERT(k <= 1024); // select limitation

  int tileSize =
    chooseTileSize<T>(
      tileSizeOverride,
      centroids.getSize(0),
      resources->getMemoryManagerCurrentDevice().getSizeAvailable());

  int maxQueriesPerIteration = std::min(tileSize, queries.getSize(0));

  // Temporary output memory space we'll use
  DeviceTensor<T, 2, true> distanceBuf1(
    mem, {maxQueriesPerIteration, centroids.getSize(0)}, defaultStream);
  DeviceTensor<T, 2, true> distanceBuf2(
    mem, {maxQueriesPerIteration, centroids.getSize(0)}, defaultStream);
  DeviceTensor<T, 2, true>* distanceBufs[2] =
    {&distanceBuf1, &distanceBuf2};

  auto streams = resources->getAlternateStreamsCurrentDevice();
  streamWait(streams, {defaultStream});

  int curStream = 0;

  for (int i = 0; i < queries.getSize(0); i += maxQueriesPerIteration) {
    int numQueriesForIteration = std::min(maxQueriesPerIteration,
                                          queries.getSize(0) - i);

    auto distanceBufView =
      distanceBufs[curStream]->narrowOutermost(0, numQueriesForIteration);
    auto queryView =
      queries.narrowOutermost(i, numQueriesForIteration);
    auto outDistanceView =
      outDistances.narrowOutermost(i, numQueriesForIteration);
    auto outIndexView =
      outIndices.narrowOutermost(i, numQueriesForIteration);

    // (query id x dim) x (centroid id, dim)' = (query id, centroid id)
    runMatrixMult(distanceBufView, false,
                  queryView, false,
                  centroidsTransposed ? *centroidsTransposed : centroids,
                  centroidsTransposed ? false : true,
                  1.0f, 0.0f, useHgemm,
                  resources->getBlasHandleCurrentDevice(),
                  streams[curStream]);

    // top-k of dot products
    // (query id, top k centroids)
    runBlockSelect(distanceBufView,
                 outDistanceView,
                 outIndexView,
                 true, k, streams[curStream]);

    curStream = (curStream + 1) % 2;
  }

  streamWait({defaultStream}, streams);
}

//
// Instantiations of the distance templates
//

void
runIPDistance(GpuResources* resources,
              Tensor<float, 2, true>& vectors,
              Tensor<float, 2, true>* vectorsTransposed,
              Tensor<float, 2, true>& queries,
              int k,
              Tensor<float, 2, true>& outDistances,
              Tensor<int, 2, true>& outIndices,
              int tileSizeOverride) {
  runIPDistance<float>(resources,
                       vectors,
                       vectorsTransposed,
                       queries,
                       k,
                       outDistances,
                       outIndices,
                       false,
                       tileSizeOverride);
}

#ifdef FAISS_USE_FLOAT16
void
runIPDistance(GpuResources* resources,
              Tensor<half, 2, true>& vectors,
              Tensor<half, 2, true>* vectorsTransposed,
              Tensor<half, 2, true>& queries,
              int k,
              Tensor<half, 2, true>& outDistances,
              Tensor<int, 2, true>& outIndices,
              bool useHgemm,
              int tileSizeOverride) {
  runIPDistance<half>(resources,
                      vectors,
                      vectorsTransposed,
                      queries,
                      k,
                      outDistances,
                      outIndices,
                      useHgemm,
                      tileSizeOverride);
}
#endif
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
              int tileSizeOverride) {
  runL2DistanceWithGraph<float>(resources,
                       vectors,
                       vectorsTransposed,
                       vectorNorms,
                       queries,
                       k,
                       k1,
                      graphIndices,
                      graphDists,
                      outDistances,
                      outGraphDistances,
                       outIndices,
                      outDistances2nd,
                      outIndices2nd,
                       begin ,end,
                       false,
                       ignoreOutDistances,
                       tileSizeOverride);
}


void runL2DistanceWithGraph(GpuResources* resources,
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
              int tileSizeOverride) {
  runL2DistanceWithGraph<float>(resources,
                       vectors,
                       vectorsTransposed,
                       vectorNorms,
                       queries,
                       k,
                      graphIndices,
                      outDistances,
                      outGraphDistances,
                       outIndices,
                       false,
                       ignoreOutDistances,
                       tileSizeOverride);
}

#ifdef FAISS_USE_FLOAT16
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
              int tileSizeOverride) {
  runL2DistanceWithGraph<half>(resources,
                      vectors,
                      vectorsTransposed,
                      vectorNorms,
                      queries,
                      k,
                      graphIndices,
                      outDistances,
                      outGraphDistances,
                      outIndices,
                      useHgemm,
                      ignoreOutDistances,
                      tileSizeOverride);
}
#endif




void
runL2Distance(GpuResources* resources,
              Tensor<float, 2, true>& vectors,
              Tensor<float, 2, true>* vectorsTransposed,
              Tensor<float, 1, true>* vectorNorms,
              Tensor<float, 2, true>& queries,
              int k,
              Tensor<float, 2, true>& outDistances,
              Tensor<int, 2, true>& outIndices,
              bool ignoreOutDistances,
              int tileSizeOverride) {

  runL2Distance<float>(resources,
                       vectors,
                       vectorsTransposed,
                       vectorNorms,
                       queries,
                       k,
                       outDistances,
                       outIndices,
                       false,
                       ignoreOutDistances,
                       tileSizeOverride);
}

#ifdef FAISS_USE_FLOAT16
void
runL2Distance(GpuResources* resources,
              Tensor<half, 2, true>& vectors,
              Tensor<half, 2, true>* vectorsTransposed,
              Tensor<half, 1, true>* vectorNorms,
              Tensor<half, 2, true>& queries,
              int k,
              Tensor<half, 2, true>& outDistances,
              Tensor<int, 2, true>& outIndices,
              bool useHgemm,
              bool ignoreOutDistances,
              int tileSizeOverride) {
  runL2Distance<half>(resources,
                      vectors,
                      vectorsTransposed,
                      vectorNorms,
                      queries,
                      k,
                      outDistances,
                      outIndices,
                      useHgemm,
                      ignoreOutDistances,
                      tileSizeOverride);
}
#endif

} } // namespace
