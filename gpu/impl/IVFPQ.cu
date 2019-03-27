/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "IVFPQ.cuh"
#include "../GpuResources.h"
#include "BroadcastSum.cuh"
#include "Distance.cuh"
#include "FlatIndex.cuh"
#include "InvertedListAppend.cuh"
#include "L2Norm.cuh"
#include "PQCodeDistances.cuh"
#include "PQScanMultiPassNoPrecomputed.cuh"
#include "PQScanMultiPassPrecomputed.cuh"
#include "RemapIndices.h"
#include "VectorResidual.cuh"
#include "../utils/DeviceDefs.cuh"
#include "./Distance.cuh"
#include "../utils/helper.cuh"
#include "../utils/CopyUtils.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/HostTensor.cuh"
#include "../utils/MatrixMult.cuh"
#include "../utils/NoTypeTensor.cuh"
#include "../utils/Transpose.cuh"
#include "../utils/ConversionOperators.cuh"
#include "../utils/LoadStoreOperators.cuh"
#include "../utils/MathOperators.cuh"
#include "PQCodeLoad.cuh"
#include <limits>
#include <thrust/host_vector.h>
#include <unordered_map>

namespace faiss { namespace gpu {

IVFPQ::IVFPQ(GpuResources* resources,
             FlatIndex* quantizer,
             int numSubQuantizers,
             int bitsPerSubQuantizer,
             float* pqCentroidData,
             IndicesOptions indicesOptions,
             bool useFloat16LookupTables,
             MemorySpace space) :
    IVFBase(resources,
            quantizer,
            numSubQuantizers,
            indicesOptions,
            space),
    numSubQuantizers_(numSubQuantizers),
    bitsPerSubQuantizer_(bitsPerSubQuantizer),
    numSubQuantizerCodes_(utils::pow2(bitsPerSubQuantizer_)),
    dimPerSubQuantizer_(dim_ / numSubQuantizers),
    precomputedCodes_(false),
    useFloat16LookupTables_(useFloat16LookupTables) {
  FAISS_ASSERT(pqCentroidData);

  FAISS_ASSERT(bitsPerSubQuantizer_ <= 8);
  FAISS_ASSERT(dim_ % numSubQuantizers_ == 0);
  FAISS_ASSERT(isSupportedPQCodeLength(bytesPerVector_));

#ifndef FAISS_USE_FLOAT16
  FAISS_ASSERT(!useFloat16LookupTables_);
#endif

  setPQCentroids_(pqCentroidData);
}

IVFPQ::IVFPQ(GpuResources* resources,
             FlatIndex* quantizer,
             int numedge,
             int numSubQuantizers,
             int bitsPerSubQuantizer,
             float* pqCentroidData,
             IndicesOptions indicesOptions,
             bool useFloat16LookupTables,
             MemorySpace space) :
    IVFBase(resources,
            quantizer,
            numedge,
            numSubQuantizers,
            indicesOptions,
            space),
    numSubQuantizers_(numSubQuantizers),
    bitsPerSubQuantizer_(bitsPerSubQuantizer),
    numSubQuantizerCodes_(utils::pow2(bitsPerSubQuantizer_)),
    dimPerSubQuantizer_(dim_ / numSubQuantizers),
    precomputedCodes_(false),
    useFloat16LookupTables_(useFloat16LookupTables) {
  FAISS_ASSERT(pqCentroidData);

  FAISS_ASSERT(bitsPerSubQuantizer_ <= 8);
  FAISS_ASSERT(dim_ % numSubQuantizers_ == 0);
  FAISS_ASSERT(isSupportedPQCodeLength(bytesPerVector_));

#ifndef FAISS_USE_FLOAT16
  FAISS_ASSERT(!useFloat16LookupTables_);
#endif
  setPQCentroids_(pqCentroidData);
}

IVFPQ::IVFPQ(GpuResources* resources,
             FlatIndex* quantizer,
             int numedge,
             int numSubQuantizers,
             int bitsPerSubQuantizer,
             float* pqCentroidData,
             bool fromFile,
             IndicesOptions indicesOptions,
             bool useFloat16LookupTables,
             MemorySpace space) :
    IVFBase(resources,
            quantizer,
            numedge,
            numSubQuantizers,
            indicesOptions,
            space),
    numSubQuantizers_(numSubQuantizers),
    bitsPerSubQuantizer_(bitsPerSubQuantizer),
    numSubQuantizerCodes_(utils::pow2(bitsPerSubQuantizer_)),
    dimPerSubQuantizer_(dim_ / numSubQuantizers),
    precomputedCodes_(false),
    useFloat16LookupTables_(useFloat16LookupTables) {
  FAISS_ASSERT(pqCentroidData);

  FAISS_ASSERT(bitsPerSubQuantizer_ <= 8);
  FAISS_ASSERT(dim_ % numSubQuantizers_ == 0);
  FAISS_ASSERT(isSupportedPQCodeLength(bytesPerVector_));

#ifndef FAISS_USE_FLOAT16
  FAISS_ASSERT(!useFloat16LookupTables_);
#endif
  if(!fromFile)
  setPQCentroids_(pqCentroidData);
  else
  setPQCentroidsFile_(pqCentroidData);
}


IVFPQ::~IVFPQ() {
}


bool
IVFPQ::isSupportedPQCodeLength(int size) {
  switch (size) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 8:
    case 12:
    case 16:
    case 20:
    case 24:
    case 28:
    case 32:
    case 40:
    case 48:
    case 56: // only supported with float16
    case 64: // only supported with float16
    case 96: // only supported with float16
      return true;
    default:
      return false;
  }
}

bool
IVFPQ::isSupportedNoPrecomputedSubDimSize(int dims) {
  return faiss::gpu::isSupportedNoPrecomputedSubDimSize(dims);
}

void
IVFPQ::setPrecomputedCodes(bool enable) {
  if (precomputedCodes_ != enable) {
    precomputedCodes_ = enable;

    if (precomputedCodes_) {
      precomputeCodes_();
    } else {
      // Clear out old precomputed code data
      precomputedCode_ = std::move(DeviceTensor<float, 3, true>());

#ifdef FAISS_USE_FLOAT16
      precomputedCodeHalf_ = std::move(DeviceTensor<half, 3, true>());
#endif
    }
  }
}

int
IVFPQ::classifyAndAddVectors(Tensor<float, 2, true>& vecs,
                             Tensor<long, 1, true>& indices) {
  FAISS_ASSERT(vecs.getSize(0) == indices.getSize(0));
  FAISS_ASSERT(vecs.getSize(1) == dim_);

  FAISS_ASSERT(!quantizer_->getUseFloat16());
  auto& coarseCentroids = quantizer_->getVectorsFloat32Ref();
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // Number of valid vectors that we actually add; we return this
  int numAdded = 0;

  // We don't actually need this
  DeviceTensor<float, 2, true> listDistance(mem, {vecs.getSize(0), 1}, stream);
  // We use this
  DeviceTensor<int, 2, true> listIds2d(mem, {vecs.getSize(0), 1}, stream);
  auto listIds = listIds2d.view<1>({vecs.getSize(0)});

  quantizer_->query(vecs, 1, listDistance, listIds2d, false);

  // Copy the lists that we wish to append to back to the CPU
  // FIXME: really this can be into pinned memory and a true async
  // copy on a different stream; we can start the copy early, but it's
  // tiny
  HostTensor<int, 1, true> listIdsHost(listIds, stream);
   for(int i=0;i<10;i++){
    printf("listIdsHost[%d]:%d\n",i,listIdsHost[i]);
  }
  // Calculate the residual for each closest centroid
  DeviceTensor<float, 2, true> residuals(
    mem, {vecs.getSize(0), vecs.getSize(1)}, stream);

  runCalcResidual(vecs, coarseCentroids, listIds, residuals, stream);

  // Residuals are in the form
  // (vec x numSubQuantizer x dimPerSubQuantizer)
  // transpose to
  // (numSubQuantizer x vec x dimPerSubQuantizer)
  auto residualsView = residuals.view<3>(
    {residuals.getSize(0), numSubQuantizers_, dimPerSubQuantizer_});

  DeviceTensor<float, 3, true> residualsTranspose(
    mem,
    {numSubQuantizers_, residuals.getSize(0), dimPerSubQuantizer_},
    stream);

  runTransposeAny(residualsView, 0, 1, residualsTranspose, stream);

  // Get the product quantizer centroids in the form
  // (numSubQuantizer x numSubQuantizerCodes x dimPerSubQuantizer)
  // which is pqCentroidsMiddleCode_

  // We now have a batch operation to find the top-1 distances:
  // batch size: numSubQuantizer
  // centroids: (numSubQuantizerCodes x dimPerSubQuantizer)
  // residuals: (vec x dimPerSubQuantizer)
  // => (numSubQuantizer x vec x 1)

  DeviceTensor<float, 3, true> closestSubQDistance(
    mem, {numSubQuantizers_, residuals.getSize(0), 1}, stream);
  DeviceTensor<int, 3, true> closestSubQIndex(
    mem, {numSubQuantizers_, residuals.getSize(0), 1}, stream);

  for (int subQ = 0; subQ < numSubQuantizers_; ++subQ) {
    auto closestSubQDistanceView = closestSubQDistance[subQ].view();
    auto closestSubQIndexView = closestSubQIndex[subQ].view();

    auto pqCentroidsMiddleCodeView = pqCentroidsMiddleCode_[subQ].view();
    auto residualsTransposeView = residualsTranspose[subQ].view();

    runL2Distance(resources_,
                  pqCentroidsMiddleCodeView,
                  nullptr, // no transposed storage
                  nullptr, // no precomputed norms
                  residualsTransposeView,
                  1,
                  closestSubQDistanceView,
                  closestSubQIndexView,
                  // We don't care about distances
                  true,
                  // Much larger tile size, since these vectors are a
                  // lot smaller than query vectors
                  1024);
  }

  // Now, we have the nearest sub-q centroid for each slice of the
  // residual vector.
  auto closestSubQIndexView = closestSubQIndex.view<2>(
    {numSubQuantizers_, residuals.getSize(0)});

  // Transpose this for easy use
  DeviceTensor<int, 2, true> encodings(
    mem, {residuals.getSize(0), numSubQuantizers_}, stream);

  runTransposeAny(closestSubQIndexView, 0, 1, encodings, stream);

  // Now we add the encoded vectors to the individual lists
  // First, make sure that there is space available for adding the new
  // encoded vectors and indices

  // list id -> # being added
  std::unordered_map<int, int> assignCounts;

  // vector id -> offset in list
  // (we already have vector id -> list id in listIds)
  HostTensor<int, 1, true> listOffsetHost({listIdsHost.getSize(0)});

  for (int i = 0; i < listIdsHost.getSize(0); ++i) {
    int listId = listIdsHost[i];
    // printf("listId: %d\n",listId);
    // Add vector could be invalid (contains NaNs etc)
    if (listId < 0) {
      listOffsetHost[i] = -1;
      continue;
    }

    FAISS_ASSERT(listId < numLists_);
    ++numAdded;

    int offset = deviceListData_[listId]->size() / bytesPerVector_;

    auto it = assignCounts.find(listId);
    if (it != assignCounts.end()) {
      offset += it->second;
      it->second++;
    } else {
      assignCounts[listId] = 1;
    }

    listOffsetHost[i] = offset;
  }

  // If we didn't add anything (all invalid vectors), no need to
  // continue
  if (numAdded == 0) {
    return 0;
  }

  // We need to resize the data structures for the inverted lists on
  // the GPUs, which means that they might need reallocation, which
  // means that their base address may change. Figure out the new base
  // addresses, and update those in a batch on the device
  {
    // Resize all of the lists that we are appending to
    for (auto& counts : assignCounts) {
      auto& codes = deviceListData_[counts.first];
      codes->resize(codes->size() + counts.second * bytesPerVector_,
                    stream);
      int newNumVecs = (int) (codes->size() / bytesPerVector_);

      auto& indices = deviceListIndices_[counts.first];
      if ((indicesOptions_ == INDICES_32_BIT) ||
          (indicesOptions_ == INDICES_64_BIT)) {
        size_t indexSize =
          (indicesOptions_ == INDICES_32_BIT) ? sizeof(int) : sizeof(long);

        indices->resize(indices->size() + counts.second * indexSize, stream);
      } else if (indicesOptions_ == INDICES_CPU) {
        // indices are stored on the CPU side
        FAISS_ASSERT(counts.first < listOffsetToUserIndex_.size());

        auto& userIndices = listOffsetToUserIndex_[counts.first];
        userIndices.resize(newNumVecs);
      } else {
        // indices are not stored on the GPU or CPU side
        FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
      }

      // This is used by the multi-pass query to decide how much scratch
      // space to allocate for intermediate results
      maxListLength_ = std::max(maxListLength_, newNumVecs);
    }

    // Update all pointers and sizes on the device for lists that we
    // appended to
    {
      std::vector<int> listIds(assignCounts.size());
      int i = 0;
      for (auto& counts : assignCounts) {
        listIds[i++] = counts.first;
      }

      updateDeviceListInfo_(listIds, stream);
    }
  }

  // If we're maintaining the indices on the CPU side, update our
  // map. We already resized our map above.
  if (indicesOptions_ == INDICES_CPU) {
    // We need to maintain the indices on the CPU side
    HostTensor<long, 1, true> hostIndices(indices, stream);

    for (int i = 0; i < hostIndices.getSize(0); ++i) {
      int listId = listIdsHost[i];

      // Add vector could be invalid (contains NaNs etc)
      if (listId < 0) {
        continue;
      }

      int offset = listOffsetHost[i];

      FAISS_ASSERT(listId < listOffsetToUserIndex_.size());
      auto& userIndices = listOffsetToUserIndex_[listId];

      FAISS_ASSERT(offset < userIndices.size());
      userIndices[offset] = hostIndices[i];
    }
  }

  // We similarly need to actually append the new encoded vectors
  {
    DeviceTensor<int, 1, true> listOffset(mem, listOffsetHost, stream);

    // This kernel will handle appending each encoded vector + index to
    // the appropriate list
    runIVFPQInvertedListAppend(listIds,
                               listOffset,
                               encodings,
                               indices,
                               deviceListDataPointers_,
                               deviceListIndexPointers_,
                               indicesOptions_,
                               stream);
  }

  return numAdded;
}

void
IVFPQ::addCodeVectorsFromCpu(int listId,
                             const void* codes,
                             const long* indices,
                             size_t numVecs) {
  // This list must already exist
  FAISS_ASSERT(listId < deviceListData_.size());
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // If there's nothing to add, then there's nothing we have to do
  if (numVecs == 0) {
    return;
  }

  size_t lengthInBytes = numVecs * bytesPerVector_;

  auto& listCodes = deviceListData_[listId];
  auto prevCodeData = listCodes->data();

  // We only have int32 length representations on the GPU per each
  // list; the length is in sizeof(char)
  FAISS_ASSERT(listCodes->size() % bytesPerVector_ == 0);
  FAISS_ASSERT(listCodes->size() + lengthInBytes <=
               (size_t) std::numeric_limits<int>::max());

  listCodes->append((unsigned char*) codes,
                    lengthInBytes,
                    stream,
                    true /* exact reserved size */);

  // Handle the indices as well
  addIndicesFromCpu_(listId, indices, numVecs);

  // This list address may have changed due to vector resizing, but
  // only bother updating it on the device if it has changed
  if (prevCodeData != listCodes->data()) {
    deviceListDataPointers_[listId] = listCodes->data();
  }

  // And our size has changed too
  int listLength = listCodes->size() / bytesPerVector_;
    deviceListLengths_[listId] = listLength;

  // We update this as well, since the multi-pass algorithm uses it
  maxListLength_ = std::max(maxListLength_, listLength);

  // device_vector add is potentially happening on a different stream
  // than our default stream
  if (resources_->getDefaultStreamCurrentDevice() != 0) {
    streamWait({stream}, {0});
  }
}



void
IVFPQ::addLambdasFromCpu(int listId,
                             const void* lambdas,
                             size_t numVecs) {
  // This list must already exist
  FAISS_ASSERT(listId < deviceListLambdas_.size());
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // If there's nothing to add, then there's nothing we have to do
  if (numVecs == 0) {
    return;
  }

  size_t lengthInBytes = numVecs ;

  auto& listLas = deviceListLambdas_[listId];
  auto prevLaData = listLas->data();

  // We only have int32 length representations on the GPU per each
  // list; the length is in sizeof(char)
  //FAISS_ASSERT(listCodes->size() % bytesPerVector_ == 0);
 // FAISS_ASSERT(listCodes->size() + lengthInBytes <=
    //           (size_t) std::numeric_limits<int>::max());

  listLas->append((unsigned char*) lambdas,
                    lengthInBytes,
                    stream,
                    true /* exact reserved size */);


  // This list address may have changed due to vector resizing, but
  // only bother updating it on the device if it has changed
  if (prevLaData != listLas->data()) {
    deviceListLambdaPointers_[listId] = listLas->data();
  }


  // device_vector add is potentially happening on a different stream
  // than our default stream
  if (resources_->getDefaultStreamCurrentDevice() != 0) {
    streamWait({stream}, {0});
  }
}

void
IVFPQ::setPQCentroids_(float* data) {
  size_t pqSize =
    numSubQuantizers_ * numSubQuantizerCodes_ * dimPerSubQuantizer_;

  // Make sure the data is on the host
  // FIXME: why are we doing this?
  thrust::host_vector<float> hostMemory;
  hostMemory.insert(hostMemory.end(), data, data + pqSize);

  HostTensor<float, 3, true> pqHost(
    hostMemory.data(),
    {numSubQuantizers_, numSubQuantizerCodes_, dimPerSubQuantizer_});
  DeviceTensor<float, 3, true> pqDevice(
    pqHost,
    resources_->getDefaultStreamCurrentDevice());

  DeviceTensor<float, 3, true> pqDeviceTranspose(
    {numSubQuantizers_, dimPerSubQuantizer_, numSubQuantizerCodes_});
  runTransposeAny(pqDevice, 1, 2, pqDeviceTranspose,
                  resources_->getDefaultStreamCurrentDevice());

  pqCentroidsInnermostCode_ = std::move(pqDeviceTranspose);

  // Also maintain the PQ centroids in the form
  // (sub q)(code id)(sub dim)
  DeviceTensor<float, 3, true> pqCentroidsMiddleCode(
    {numSubQuantizers_, numSubQuantizerCodes_, dimPerSubQuantizer_});
  runTransposeAny(pqCentroidsInnermostCode_, 1, 2, pqCentroidsMiddleCode,
                  resources_->getDefaultStreamCurrentDevice());

  pqCentroidsMiddleCode_ = std::move(pqCentroidsMiddleCode);
}

void
IVFPQ::setPQCentroidsFile_(float* data) {
  size_t pqSize =
    numSubQuantizers_ * numSubQuantizerCodes_ * dimPerSubQuantizer_;

  // Make sure the data is on the host
  // FIXME: why are we doing this?
  thrust::host_vector<float> hostMemory;
  hostMemory.insert(hostMemory.end(), data, data + pqSize);

  HostTensor<float, 3, true> pqHost(
    hostMemory.data(),
    {numSubQuantizerCodes_,numSubQuantizers_, dimPerSubQuantizer_});
  DeviceTensor<float, 3, true> pqDevice(
    pqHost,
    resources_->getDefaultStreamCurrentDevice());
  DeviceTensor<float, 3, true> pqDeviceTransposeMiddle(
    {numSubQuantizers_, numSubQuantizerCodes_, dimPerSubQuantizer_});
  runTransposeAny(pqDevice, 0, 1, pqDeviceTransposeMiddle,
                  resources_->getDefaultStreamCurrentDevice());
  DeviceTensor<float, 3, true> pqDeviceTranspose(
    {numSubQuantizers_, dimPerSubQuantizer_, numSubQuantizerCodes_});
  runTransposeAny(pqDeviceTransposeMiddle, 1, 2, pqDeviceTranspose,
                  resources_->getDefaultStreamCurrentDevice());

  pqCentroidsInnermostCode_ = std::move(pqDeviceTranspose);

  // Also maintain the PQ centroids in the form
  // (sub q)(code id)(sub dim)
  DeviceTensor<float, 3, true> pqCentroidsMiddleCode(
    {numSubQuantizers_, numSubQuantizerCodes_, dimPerSubQuantizer_});
  runTransposeAny(pqCentroidsInnermostCode_, 1, 2, pqCentroidsMiddleCode,
                  resources_->getDefaultStreamCurrentDevice());

  pqCentroidsMiddleCode_ = std::move(pqCentroidsMiddleCode);
}


void
IVFPQ::precomputeCodes_() {
  //
  //    d = || x - y_C ||^2 + || y_R ||^2 + 2 * (y_C|y_R) - 2 * (x|y_R)
  //        ---------------   ---------------------------       -------
  //            term 1                 term 2                   term 3
  //

  // Terms 1 and 3 are available only at query time. We compute term 2
  // here.
  FAISS_ASSERT(!quantizer_->getUseFloat16());
  auto& coarseCentroids = quantizer_->getVectorsFloat32Ref();
   printf("precomputeCodes_111\n");
  // Compute ||y_R||^2 by treating
  // (sub q)(code id)(sub dim) as (sub q * code id)(sub dim)
  auto pqCentroidsMiddleCodeView =
    pqCentroidsMiddleCode_.view<2>(
      {numSubQuantizers_ * numSubQuantizerCodes_, dimPerSubQuantizer_});
  DeviceTensor<float, 1, true> subQuantizerNorms(
    {numSubQuantizers_ * numSubQuantizerCodes_});

  runL2Norm(pqCentroidsMiddleCodeView, subQuantizerNorms, true,
            resources_->getDefaultStreamCurrentDevice());
  // Compute 2 * (y_C|y_R) via batch matrix multiplication
  // batch size (sub q) x {(centroid id)(sub dim) x (code id)(sub dim)'}
  //         => (sub q) x {(centroid id)(code id)}
  //         => (sub q)(centroid id)(code id)

  // View (centroid id)(dim) as
  //      (centroid id)(sub q)(dim)
  // Transpose (centroid id)(sub q)(sub dim) to
  //           (sub q)(centroid id)(sub dim)
  auto centroidView = coarseCentroids.view<3>(
    {coarseCentroids.getSize(0), numSubQuantizers_, dimPerSubQuantizer_});
  DeviceTensor<float, 3, true> centroidsTransposed(
    {numSubQuantizers_, coarseCentroids.getSize(0), dimPerSubQuantizer_});

  runTransposeAny(centroidView, 0, 1, centroidsTransposed,
                  resources_->getDefaultStreamCurrentDevice());

  DeviceTensor<float, 3, true> coarsePQProduct(
    {numSubQuantizers_, coarseCentroids.getSize(0), numSubQuantizerCodes_});

  runIteratedMatrixMult(coarsePQProduct, false,
                        centroidsTransposed, false,
                        pqCentroidsMiddleCode_, true,
                        2.0f, 0.0f,
                        resources_->getBlasHandleCurrentDevice(),
                        resources_->getDefaultStreamCurrentDevice());
  // Transpose (sub q)(centroid id)(code id) to
  //           (centroid id)(sub q)(code id)
  DeviceTensor<float, 3, true> coarsePQProductTransposed(
    {coarseCentroids.getSize(0), numSubQuantizers_, numSubQuantizerCodes_});
  runTransposeAny(coarsePQProduct, 0, 1, coarsePQProductTransposed,
                  resources_->getDefaultStreamCurrentDevice());
  // View (centroid id)(sub q)(code id) as
  //      (centroid id)(sub q * code id)
  auto coarsePQProductTransposedView = coarsePQProductTransposed.view<2>(
    {coarseCentroids.getSize(0), numSubQuantizers_ * numSubQuantizerCodes_});



  // Sum || y_R ||^2 + 2 * (y_C|y_R)
  // i.e., add norms                              (sub q * code id)
  // along columns of inner product  (centroid id)(sub q * code id)
  runSumAlongColumnsGraph1(subQuantizerNorms, coarsePQProductTransposedView,
                   resources_->getDefaultStreamCurrentDevice());

#ifdef FAISS_USE_FLOAT16
  if (useFloat16LookupTables_) {

    precomputedCodeHalf_ = toHalf(resources_,
                                  resources_->getDefaultStreamCurrentDevice(),
                                  coarsePQProductTransposed);
    subQuantizerNormsHalf_= toHalf(resources_,
                                  resources_->getDefaultStreamCurrentDevice(),
                                  subQuantizerNorms);
  subQuantizerNorms_ = std::move(subQuantizerNorms);
    return;
  }
#endif
  // We added into the view, so `coarsePQProductTransposed` is now our
  // precomputed term 2.
  precomputedCode_ = std::move(coarsePQProductTransposed);
   subQuantizerNorms_ = std::move(subQuantizerNorms);
}
void
IVFPQ::queryGraph(Tensor<float, 2, true>& queries,
             Tensor<int, 2, true>&  devEdgeInfo,
             Tensor<float, 2, true>&  devEdgeDistInfo,
             Tensor<float, 1, true>&  devLambdaInfo,
             Tensor<float, 1, true>&  devConstInfo,
             int nprobe,int w1,
             int k,int begin ,int end,
             Tensor<float, 2, true>& outDistances,
             Tensor<long, 2, true>& outIndices) {
  // Validate these at a top level
  FAISS_ASSERT(nprobe <= 1024);
  FAISS_ASSERT(k <= 1024);
  int nedge = devEdgeInfo.getSize(1);
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStreamCurrentDevice();
  nprobe = std::min(nprobe, quantizer_->getSize());

  FAISS_ASSERT(queries.getSize(1) == dim_);
  FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

  // Reserve space for the closest coarse centroids
  DeviceTensor<float, 2, true>
    coarseDistances(mem, {queries.getSize(0), nprobe}, stream);
  DeviceTensor<int, 2, true>
    coarseIndices(mem, {queries.getSize(0), nprobe}, stream);
  // Reserve space for the closest coarse centroids
  DeviceTensor<float, 2, true>
    coarseDistances2nd(mem, {queries.getSize(0), w1}, stream);
  DeviceTensor<int, 2, true>
    coarseIndices2nd(mem, {queries.getSize(0), w1}, stream);

  // Find the `nprobe` closest coarse centroids; we can use int
  // indices both internally and externally
    DeviceTensor<float, 2, true>
    graphDistances(mem, {queries.getSize(0), w1}, stream);




    quantizer_->query(queries, nprobe,w1,devEdgeInfo,devEdgeDistInfo,coarseDistances,
                 graphDistances,coarseIndices,coarseDistances2nd,coarseIndices2nd, begin , end, true);
    //outputVec("coarseDistances2nd",coarseDistances2nd.data(),1024,stream);
  // outputVec("graphDistances",graphDistances.data(),100,stream);
    //outputVecInt("coarseIndices",coarseIndices.data(),10,stream);
    //outputVecInt("coarseIndices2nd",coarseIndices2nd.data(),100,stream);

   if (precomputedCodes_) {
     printf("precomputedCodes_,k:%d\n",k);
    runPQPrecomputedCodesGraph_(queries,
                           coarseDistances,
                           coarseIndices,
                           graphDistances,
                           devEdgeInfo,
                           devEdgeDistInfo,
                           devLambdaInfo,
                           devConstInfo,
                           k,
                           coarseDistances2nd,
                           coarseIndices2nd,
                           outDistances,
                           outIndices);
  } else {
  printf("preNOcomputedCodes_,k:%d\n",k);
    runPQNoPrecomputedCodes_(queries,
                             coarseDistances,
                             coarseIndices,
                             k,
                             outDistances,
                             outIndices);
  }

  // If the GPU isn't storing indices (they are on the CPU side), we
  // need to perform the re-mapping here
  // FIXME: we might ultimately be calling this function with inputs
  // from the CPU, these are unnecessary copies
  if (indicesOptions_ == INDICES_CPU) {
    HostTensor<long, 2, true> hostOutIndices(outIndices, stream);

    ivfOffsetToUserIndex(hostOutIndices.data(),
                         numLists_,
                         hostOutIndices.getSize(0),
                         hostOutIndices.getSize(1),
                         listOffsetToUserIndex_);

    // Copy back to GPU, since the input to this function is on the
    // GPU
    outIndices.copyFrom(hostOutIndices, stream);
  }
}

void
IVFPQ::queryGraph1(Tensor<float, 2, true>& queries,
             Tensor<int, 2, true>&  devEdgeInfo,
             Tensor<float, 2, true>&  devEdgeDistInfo,
             Tensor<float, 1, true>&  devLambdaInfo,
             Tensor<float, 1, true>&  devConstInfo,
             int nprobe,
             int w1,
             int k,int begin ,int end,
             long* outIndices) {
  // Validate these at a top level
 // FAISS_ASSERT(k <= 1024);
  int nedge = devEdgeInfo.getSize(1);
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStreamCurrentDevice();
  nprobe = std::min(nprobe, quantizer_->getSize());

  FAISS_ASSERT(queries.getSize(1) == dim_);

  // Reserve space for the closest coarse centroids
  DeviceTensor<float, 2, true>
    coarseDistances(mem, {queries.getSize(0), nprobe}, stream);
  DeviceTensor<int, 2, true>
    coarseIndices(mem, {queries.getSize(0), nprobe}, stream);
  // Reserve space for the closest coarse centroids
  DeviceTensor<float, 2, true>
    coarseDistances2nd(mem, {queries.getSize(0), w1}, stream);
  DeviceTensor<int, 2, true>
    coarseIndices2nd(mem, {queries.getSize(0), w1}, stream);

  // Find the `nprobe` closest coarse centroids; we can use int
  // indices both internally and externally
    DeviceTensor<float, 2, true>
    graphDistances(mem, {queries.getSize(0), w1}, stream);




    quantizer_->query(queries, nprobe,w1,devEdgeInfo,devEdgeDistInfo,coarseDistances,
                 graphDistances,coarseIndices,coarseDistances2nd,coarseIndices2nd, begin , end,true);
    //outputVec("coarseDistances2nd",coarseDistances2nd.data(),1024,stream);
  // outputVec("graphDistances",graphDistances.data(),100,stream);
    //outputVecInt("coarseIndices",coarseIndices.data(),10,stream);
    //outputVecInt("coarseIndices2nd",coarseIndices2nd.data()+1024,1024,stream);


  // If the GPU isn't storing indices (they are on the CPU side), we
  // need to perform the re-mapping here
  // FIXME: we might ultimately be calling this function with inputs
  // from the CPU, these are unnecessary copies
  if (indicesOptions_ == INDICES_CPU) {

    HostTensor<int, 2, true> hostIndices2nd(coarseIndices2nd, stream);
    int queriesNum=queries.getSize(0);
    printf("queriesNum= %d\n", queriesNum);
      for (int q = 0; q < queriesNum; ++q) {
        int ncode= 0;
         if(q%100==0)
          printf("q = %d\n",   q);
        for (int r = 0; r < hostIndices2nd.getSize(1); ++r) {

          int listId = hostIndices2nd[q][r];
          auto& listIndices = listOffsetToUserIndex_[listId];

          //FAISS_ASSERT(listOffset < listIndices.size());
          for(int j =0;j<listIndices.size();j++){
            if(ncode+j<k){
                long index = q*k;
                index +=ncode+j;
               // if(q==780)
               // printf("index = %ld\n",   index);
                outIndices[index] = listIndices[j];
            }

            else
             break;

          }
            ncode +=listIndices.size();
            if(ncode>=k)
            break;
        }
      }

  }
}

void
IVFPQ::testGraph(Tensor<float, 2, true>& queries,
             Tensor<int, 2, true>&  devEdgeInfo,
             Tensor<float, 2, true>&  devEdgeDistInfo,
             Tensor<float, 1, true>&  devLambdaInfo,
             int nprobe,
             int k,
             Tensor<float, 2, true>& outDistances,
             Tensor<long, 2, true>& outIndices) {
  // Validate these at a top level
  FAISS_ASSERT(nprobe <= 1024);
  FAISS_ASSERT(k <= 1024);
  int nedge = devEdgeInfo.getSize(1);
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStreamCurrentDevice();
  nprobe = std::min(nprobe, quantizer_->getSize());

  FAISS_ASSERT(queries.getSize(1) == dim_);
  FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

  // Reserve space for the closest coarse centroids
  DeviceTensor<float, 2, true>
    coarseDistances(mem, {queries.getSize(0), nprobe}, stream);
  DeviceTensor<int, 2, true>
    coarseIndices(mem, {queries.getSize(0), nprobe}, stream);

    DeviceTensor<float, 2, true>
    coarseDistances2nd(mem, {queries.getSize(0), 1024}, stream);
  DeviceTensor<int, 2, true>
    coarseIndices2nd(mem, {queries.getSize(0), nprobe*nedge}, stream);


  // Find the `nprobe` closest coarse centroids; we can use int
  // indices both internally and externally
    DeviceTensor<float, 2, true>
    graphDistances(mem, {queries.getSize(0), nprobe*nedge}, stream);
    Tensor<float, 2, true>
    subquery(queries.data(), {10, queries.getSize(1)});
    outputVec("query[0]:",subquery.data(),queries.getSize(1),stream);
     testL2DistanceWithGraph(resources_,
                     quantizer_->getVectorsFloat32Ref(),
                     nullptr,
                    &quantizer_->norms_,
                    subquery,
                    nprobe,
                    135,
                    devEdgeInfo,
                   coarseDistances,
                    graphDistances,
                    coarseIndices,
                    false,
                    false);


      // Reserve space for the closest coarse centroids
    // outputVec("coarseDistances",coarseDistances.data(),10, stream);
   // outputVecInt("coarseIndices",coarseIndices.data(),10, stream);
    // outputVec("graphDistances",graphDistances.data(),10*32, stream);

    testPQPrecomputedCodesGraph_(subquery,
                           coarseDistances,
                           coarseIndices,
                           graphDistances,
                           devEdgeInfo,
                           devEdgeDistInfo,
                           devLambdaInfo,
                           k,
                           135,
                           1,
                           outDistances,
                           outIndices);
  int length= getListLength(135*nedge+1);

  /// For debugging purposes, return the list codes of a particular
  /// list
  std::vector<unsigned char> listCodes =  getListCodes(135*nedge+1);
   std::vector<unsigned char> listLambdas = getListLambdas(135*nedge+1);

      printf ("\n listLambdas: ");
            for (int j = 0; j < length ;j++) {
                printf ("%d ", listLambdas.data()[j]);
            }
            printf ("\n");
          for (int i = 0; i < length; i++) {
            printf ("listCodes %2d: ", i);
            for (int j = 0; j < 64; j++) {
                printf ("%d ", listCodes[j + i * 64]);
            }
             printf ("\n");
          }

}


void
IVFPQ::query(Tensor<float, 2, true>& queries,
             int nprobe,
             int k,
             Tensor<float, 2, true>& outDistances,
             Tensor<long, 2, true>& outIndices) {
  // Validate these at a top level
  FAISS_ASSERT(nprobe <= 1024);
  FAISS_ASSERT(k <= 1024);

  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStreamCurrentDevice();
  nprobe = std::min(nprobe, quantizer_->getSize());

  FAISS_ASSERT(queries.getSize(1) == dim_);
  FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

  // Reserve space for the closest coarse centroids
  DeviceTensor<float, 2, true>
    coarseDistances(mem, {queries.getSize(0), nprobe}, stream);
  DeviceTensor<int, 2, true>
    coarseIndices(mem, {queries.getSize(0), nprobe}, stream);

  // Find the `nprobe` closest coarse centroids; we can use int
  // indices both internally and externally
  quantizer_->query(queries,
                    nprobe,
                    coarseDistances,
                    coarseIndices,
                    true);

  if (precomputedCodes_) {
    runPQPrecomputedCodes_(queries,
                           coarseDistances,
                           coarseIndices,
                           k,
                           outDistances,
                           outIndices);
  } else {
    runPQNoPrecomputedCodes_(queries,
                             coarseDistances,
                             coarseIndices,
                             k,
                             outDistances,
                             outIndices);
  }

  // If the GPU isn't storing indices (they are on the CPU side), we
  // need to perform the re-mapping here
  // FIXME: we might ultimately be calling this function with inputs
  // from the CPU, these are unnecessary copies
  if (indicesOptions_ == INDICES_CPU) {
    HostTensor<long, 2, true> hostOutIndices(outIndices, stream);

    ivfOffsetToUserIndex(hostOutIndices.data(),
                         numLists_,
                         hostOutIndices.getSize(0),
                         hostOutIndices.getSize(1),
                         listOffsetToUserIndex_);

    // Copy back to GPU, since the input to this function is on the
    // GPU
    outIndices.copyFrom(hostOutIndices, stream);
  }
}

std::vector<unsigned char>
IVFPQ::getListCodes(int listId) const {
  FAISS_ASSERT(listId < deviceListData_.size());

  return deviceListData_[listId]->copyToHost<unsigned char>(
    resources_->getDefaultStreamCurrentDevice());
}


std::vector<unsigned char>
IVFPQ::getListCodesCPU(int listId) const {
  FAISS_ASSERT(listId < listOffsetToUserCode_.size());

  auto& userCodes = listOffsetToUserCode_[listId];

  return userCodes;
}


std::vector<unsigned char>
IVFPQ::getListLambdas(int listId) const {
  FAISS_ASSERT(listId < deviceListLambdas_.size());

  return deviceListLambdas_[listId]->copyToHost<unsigned char>(
    resources_->getDefaultStreamCurrentDevice());
}

Tensor<float, 3, true>
IVFPQ::getPQCentroids() {
  return pqCentroidsMiddleCode_;
}



template <int NumSubQuantizers>
__global__ void
pqScanPrecomputedMultiPassGraph1(Tensor<float, 2, true> precompTerm1,
                           Tensor<float, 1, true> subQuantizerNorms,
                           Tensor<float, 3, true> precompTerm2,
                           Tensor<float, 3, true> precompTerm3,
                           int listId,
                          int iter,
                           Tensor<float, 2, true> graphDistances,
                           Tensor<int, 2, true> edgeInfo,
                           Tensor<float, 2, true> edgeDistInfo,
                           Tensor<float, 1, true> lambdaInfo,
                           void** listCodes,
                           int* listLengths,
                           void** listLambdas,
                            void** listIndices,
                          Tensor<float, 2, true> outDistances,
                          Tensor<long, 2, true> outIndices) {
  // precomputed term 2 + 3 storage
  // (sub q)(code id)
  extern __shared__ char smemTerm23[];
  float* term23 = (float*) smemTerm23;

  int nedge = edgeInfo.getSize(1);
  // Each block handles a single query
  auto queryId = blockIdx.y;
  auto probeId = blockIdx.x;
  auto codesPerSubQuantizer = precompTerm2.getSize(2);
  auto precompTermSize = precompTerm2.getSize(1) * codesPerSubQuantizer;
   float* smemIter = (float*) smemTerm23;
  float* term4 = smemIter;
      // This is where we start writing out data
      // We ensure that before the array (at offset -1), there is a 0 value
      //int outBase = *(prefixSumOffsets[queryId][probeId*nedge+iter].data() - 1);
      float* distanceOut = outDistances[queryId].data();
      long* indiceOut = outIndices[queryId].data();
      auto edgeId = edgeInfo[listId][iter];
      // Safety guard in case NaNs in input cause no list ID to be generated
      if (listId == -1) {
        return;
      }

      unsigned char* codeList = (unsigned char*) listCodes[listId*nedge+iter];

      uint8_t* lambdaList = (uint8_t*) listLambdas[listId*nedge+iter];
       long* indiceList  =(long*) listIndices[listId*nedge+iter];
      int limit = listLengths[listId*nedge+iter];

      constexpr int kNumCode32 = NumSubQuantizers <= 4 ? 1 :
        (NumSubQuantizers / 4);
      unsigned int code32[kNumCode32];
      unsigned int nextCode32[kNumCode32];
      uint8_t lambda[2];

      // We double-buffer the code loading, which improves memory utilization
      if (threadIdx.x < limit) {
        LoadCode32<NumSubQuantizers>::load(code32, codeList, threadIdx.x);
        lambda[0] =lambdaList[threadIdx.x];

      }

      // Load precomputed terms 1, 2, 3
      float term1 = precompTerm1[queryId][probeId];
      float term5 = edgeDistInfo[listId][iter];
      float term6 = graphDistances[queryId][probeId*nedge+iter];
      float* term2 = precompTerm2[listId].data();
      float* term21 = precompTerm2[edgeId].data();
      float* term3 = precompTerm3[queryId].data();
      float* norm = subQuantizerNorms.data();

     // loadPrecomputedTerm<LookupT, LookupVecT>(term23,
      ///                                         precompTerm2[listId].data(),
      //                                         precompTerm3[queryId].data(),
      //                                        precompTermSize);

      // Prevent WAR dependencies
      __syncthreads();

      // Each thread handles one code element in the list, with a
      // block-wide stride
      for (int codeIndex = threadIdx.x;
           codeIndex < limit;
           codeIndex += blockDim.x) {

        // Prefetch next codes
        if (codeIndex + blockDim.x < limit) {
          LoadCode32<NumSubQuantizers>::load(
            nextCode32, codeList, codeIndex + blockDim.x);
            lambda[1] =lambdaList[codeIndex + blockDim.x];
        }

     float la =lambdaInfo[lambda[0]];
       float dist =term1+la*(term6-term1)+(la*la-la)*term5;


        if(codeIndex==0&&queryId==0){
        printf("term6:%.4f,term1:%.4f\n",term6,term1);
        }
       float term21=0;
       float term31=0;
    #pragma unroll
        for (int word = 0; word < kNumCode32; ++word) {
          constexpr int kBytesPerCode32 =
            NumSubQuantizers < 4 ? NumSubQuantizers : 4;

          if (kBytesPerCode32 == 1) {
            auto code = code32[0];
            dist += ConvertTo<float>::to(Math<float>::add(term2[code],term3[code]));
             term21+=ConvertTo<float>::to(term2[code]);
              term31+=ConvertTo<float>::to(term3[code]);
          } else {
    #pragma unroll
            for (int byte = 0; byte < kBytesPerCode32; ++byte) {
              auto code = getByte(code32[word], byte * 8, 8);

              auto offset =
                codesPerSubQuantizer * (word * kBytesPerCode32 + byte);

              dist += ConvertTo<float>::to(Math<float>::add(term2[offset + code],term3[offset + code]));
              dist += norm[offset + code];
              term21+=ConvertTo<float>::to(term2[offset + code])+norm[offset + code];
              term31+=ConvertTo<float>::to(term3[offset + code]);
            // if(codeIndex==0&&queryId==0){
              //  printf("term2[%d][%d]:%.4f,",word * kBytesPerCode32 + byte,code,
             //     ConvertTo<float>::to(term2[offset + code])+norm[offset + code]);
              // }
             //  if(codeIndex==0&&queryId==0){
             //   printf("norms[%d][%d]:%.4f,",word * kBytesPerCode32 + byte,code,
             //  norm[offset + code]);
            //   }

             // if(codeIndex==0&&queryId==0){
             //   printf("term3[%d][%d]:%.4f,",word * kBytesPerCode32 + byte,code,
              //  ConvertTo<float>::to(term3[offset + code]));
             //  }
            }
          }
        }

       // if(codeIndex==0&&queryId==0){
          // printf("lambda:%.4f,term1:%.4f,term5:%.4f,term6:%.4f,term2:%.4f,term3:%.4f\n",
        //        la,term1,term5,term6,term21,term31);
        // }

        // Write out intermediate distance result
        // We do not maintain indices here, in order to reduce global
        // memory traffic. Those are recovered in the final selection step.
        distanceOut[codeIndex] = dist;
        indiceOut[codeIndex]=indiceList[codeIndex];
        // Rotate buffers
    #pragma unroll
        for (int word = 0; word < kNumCode32; ++word) {
          code32[word] = nextCode32[word];
        }
        lambda[0]=lambda[1];
      }
       __syncthreads();

      // We double-buffer the code loading, which improves memory utilization
      if (threadIdx.x < limit) {
        LoadCode32<NumSubQuantizers>::load(code32, codeList, threadIdx.x);
        lambda[0] =lambdaList[threadIdx.x];
      }
      //loadPrecomputedTermGraph<LookupT, LookupVecT>(term4,
       //                                        precompTerm2[edgeId].data(),
       //                                        precompTerm2[listId].data(),
        //                                       precompTermSize);

      // Prevent WAR dependencies
      __syncthreads();

     for (int codeIndex = threadIdx.x;
           codeIndex < limit;
           codeIndex += blockDim.x) {
        // Prefetch next codes
        if (codeIndex + blockDim.x < limit) {
          LoadCode32<NumSubQuantizers>::load(
            nextCode32, codeList, codeIndex + blockDim.x);
            lambda[1] =lambdaList[codeIndex + blockDim.x];
        }

        float la =lambdaInfo[lambda[0]];
       float dist =distanceOut[codeIndex];

    #pragma unroll
        for (int word = 0; word < kNumCode32; ++word) {
          constexpr int kBytesPerCode32 =
            NumSubQuantizers < 4 ? NumSubQuantizers : 4;

          if (kBytesPerCode32 == 1) {
            auto code = code32[0];
            dist += la*ConvertTo<float>::to(Math<float>::sub(term21[code],term2[code]));

          } else {
    #pragma unroll
            for (int byte = 0; byte < kBytesPerCode32; ++byte) {
              auto code = getByte(code32[word], byte * 8, 8);

              auto offset =
                codesPerSubQuantizer * (word * kBytesPerCode32 + byte);

              dist += la*ConvertTo<float>::to(Math<float>::sub(term21[offset + code],term2[offset + code]));
            }
          }
        }

        // Write out intermediate distance result
        // We do not maintain indices here, in order to reduce global
        // memory traffic. Those are recovered in the final selection step.
        float dist1 =(1-la)*term1+la*term6+(la*la-la)*term5;
        //distanceOut[codeIndex] = dist1;

        distanceOut[codeIndex] = dist;
        // Rotate buffers
    #pragma unroll
        for (int word = 0; word < kNumCode32; ++word) {
          code32[word] = nextCode32[word];
        }
        lambda[0]=lambda[1];
    }
}
void
IVFPQ::testPQPrecomputedCodesGraph_(
  Tensor<float, 2, true>& queries,
  DeviceTensor<float, 2, true>& coarseDistances,
  DeviceTensor<int, 2, true>& coarseIndices,
  DeviceTensor<float, 2, true>& graphDistances,
  Tensor<int, 2, true>& edgeInfo,
  Tensor<float, 2, true>& edgeDistInfo,
  Tensor<float, 1, true>& lambdaInfo,
  int k,
  int listId,
  int edgeId,
  Tensor<float, 2, true>& outDistances,
  Tensor<long, 2, true>& outIndices) {
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // Compute precomputed code term 3, - 2 * (x|y_R)
  // This is done via batch MM
  // {sub q} x {(query id)(sub dim) * (code id)(sub dim)'} =>
  // {sub q} x {(query id)(code id)}
  DeviceTensor<float, 3, true> term3Transposed(
    mem,
    {queries.getSize(0), numSubQuantizers_, numSubQuantizerCodes_},
    stream);

  // These allocations within are only temporary, so release them when
  // we're done to maximize free space
  {
    auto querySubQuantizerView = queries.view<3>(
      {queries.getSize(0), numSubQuantizers_, dimPerSubQuantizer_});
    DeviceTensor<float, 3, true> queriesTransposed(
      mem,
      {numSubQuantizers_, queries.getSize(0), dimPerSubQuantizer_},
      stream);
    runTransposeAny(querySubQuantizerView, 0, 1, queriesTransposed, stream);

    DeviceTensor<float, 3, true> term3(
      mem,
      {numSubQuantizers_, queries.getSize(0), numSubQuantizerCodes_},
      stream);

    runIteratedMatrixMult(term3, false,
                          queriesTransposed, false,
                          pqCentroidsMiddleCode_, true,
                          -2.0f, 0.0f,
                          resources_->getBlasHandleCurrentDevice(),
                          stream);

    runTransposeAny(term3, 0, 1, term3Transposed, stream);

  }

  NoTypeTensor<3, true> term2;
  NoTypeTensor<3, true> term3;

  term2 = NoTypeTensor<3, true>(precomputedCode_);
  term3 = NoTypeTensor<3, true>(term3Transposed);
    auto kThreadsPerBlock = 256;

    auto grid = dim3(coarseIndices.getSize(1),
                     coarseIndices.getSize(0));
    auto block = dim3(kThreadsPerBlock);


  // Compute ||y_R||^2 by treating
  // (sub q)(code id)(sub dim) as (sub q * code id)(sub dim)
  auto pqCentroidsMiddleCodeView =
    pqCentroidsMiddleCode_.view<2>(
      {numSubQuantizers_ * numSubQuantizerCodes_, dimPerSubQuantizer_});
  DeviceTensor<float, 1, true> subQuantizerNorms( mem,
    {numSubQuantizers_ * numSubQuantizerCodes_},stream);

  runL2Norm(pqCentroidsMiddleCodeView, subQuantizerNorms, true,
            stream);
    outputVec("subQuantizerNorms",subQuantizerNorms.data(),10,stream);

    // pq precomputed terms (2 + 3)
    auto smem = sizeof(float);
    auto term2T = term2.toTensor<float>();
      auto term3T = term3.toTensor<float>();
  pqScanPrecomputedMultiPassGraph1<64><<<grid, block, smem, stream>>>(coarseDistances,
                            subQuantizerNorms,
                            term2T,
                            term3T ,
                           listId,
                           edgeId,
                           graphDistances,
                           edgeInfo,
                           edgeDistInfo,
                           lambdaInfo,
                           deviceListDataPointers_.data().get(),
                           deviceListLengths_.data().get(),
                           deviceListLambdaPointers_.data().get(),
                           deviceListIndexPointers_.data().get(),
                            outDistances,
                            outIndices);
}

void
IVFPQ::runPQPrecomputedCodesGraph_(
  Tensor<float, 2, true>& queries,
  DeviceTensor<float, 2, true>& coarseDistances,
  DeviceTensor<int, 2, true>& coarseIndices,
  DeviceTensor<float, 2, true>& graphDistances,
  Tensor<int, 2, true>& edgeInfo,
  Tensor<float, 2, true>& edgeDistInfo,
  Tensor<float, 1, true>& lambdaInfo,
   Tensor<float, 1, true>&  constInfo,
  int k,
  DeviceTensor<float, 2, true>& coarseDistances2nd,
  DeviceTensor<int, 2, true>& coarseIndices2nd,
  Tensor<float, 2, true>& outDistances,
  Tensor<long, 2, true>& outIndices) {
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // Compute precomputed code term 3, - 2 * (x|y_R)
  // This is done via batch MM
  // {sub q} x {(query id)(sub dim) * (code id)(sub dim)'} =>
  // {sub q} x {(query id)(code id)}
  DeviceTensor<float, 3, true> term3Transposed(
    mem,
    {queries.getSize(0), numSubQuantizers_, numSubQuantizerCodes_},
    stream);

  // These allocations within are only temporary, so release them when
  // we're done to maximize free space
  {
    auto querySubQuantizerView = queries.view<3>(
      {queries.getSize(0), numSubQuantizers_, dimPerSubQuantizer_});
    DeviceTensor<float, 3, true> queriesTransposed(
      mem,
      {numSubQuantizers_, queries.getSize(0), dimPerSubQuantizer_},
      stream);
    runTransposeAny(querySubQuantizerView, 0, 1, queriesTransposed, stream);

    DeviceTensor<float, 3, true> term3(
      mem,
      {numSubQuantizers_, queries.getSize(0), numSubQuantizerCodes_},
      stream);

    runIteratedMatrixMult(term3, false,
                          queriesTransposed, false,
                          pqCentroidsMiddleCode_, true,
                          -2.0f, 0.0f,
                          resources_->getBlasHandleCurrentDevice(),
                          stream);

    runTransposeAny(term3, 0, 1, term3Transposed, stream);

  }

 NoTypeTensor<3, true> term2;
  NoTypeTensor<3, true> term3;
  // NoTypeTensor<1, true> subQuantizerNorms;
#ifdef FAISS_USE_FLOAT16
  DeviceTensor<half, 3, true> term3Half;

  if (useFloat16LookupTables_) {
    printf("useFloat16LookupTables_\n");
    term3Half = toHalf(resources_, stream, term3Transposed);
    term2 = NoTypeTensor<3, true>(precomputedCodeHalf_);
    term3 = NoTypeTensor<3, true>(term3Half);
    //subQuantizerNorms=NoTypeTensor<1, true>(subQuantizerNormsHalf_);
  }
#endif

  if (!useFloat16LookupTables_) {
   printf("useNO-Float16LookupTables_\n");
    term2 = NoTypeTensor<3, true>(precomputedCode_);
    term3 = NoTypeTensor<3, true>(term3Transposed);
    //subQuantizerNorms=NoTypeTensor<1, true>(subQuantizerNorms_);
  }




  runPQScanMultiPassPrecomputedGraph(queries,
                                 coarseDistances2nd, // term 1
                                subQuantizerNorms_,
                                term2, // term 2
                                term3, // term 3
                                coarseIndices,
                                graphDistances,
                                edgeInfo,
                                edgeDistInfo,
                                lambdaInfo,
                                constInfo,
                                useFloat16LookupTables_,
                                bytesPerVector_,
                                numSubQuantizers_,
                                numSubQuantizerCodes_,
                                deviceListDataPointers_,
                                deviceListIndexPointers_,
                                deviceListLambdaPointers_,
                                deviceListConstPointers_,
                                indicesOptions_,
                                deviceListLengths_,
                                maxListLength_,
                                k,
                                coarseIndices2nd,
                                outDistances,
                                outIndices,
                                resources_);
}




void
IVFPQ::runPQPrecomputedCodesGraph_(
  Tensor<float, 2, true>& queries,
  DeviceTensor<float, 2, true>& coarseDistances,
  DeviceTensor<int, 2, true>& coarseIndices,
  DeviceTensor<float, 2, true>& graphDistances,
  Tensor<int, 2, true>& edgeInfo,
  Tensor<float, 2, true>& edgeDistInfo,
  Tensor<float, 1, true>& lambdaInfo,
  int k,
  Tensor<float, 2, true>& outDistances,
  Tensor<long, 2, true>& outIndices) {
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // Compute precomputed code term 3, - 2 * (x|y_R)
  // This is done via batch MM
  // {sub q} x {(query id)(sub dim) * (code id)(sub dim)'} =>
  // {sub q} x {(query id)(code id)}
  DeviceTensor<float, 3, true> term3Transposed(
    mem,
    {queries.getSize(0), numSubQuantizers_, numSubQuantizerCodes_},
    stream);

  // These allocations within are only temporary, so release them when
  // we're done to maximize free space
  {
    auto querySubQuantizerView = queries.view<3>(
      {queries.getSize(0), numSubQuantizers_, dimPerSubQuantizer_});
    DeviceTensor<float, 3, true> queriesTransposed(
      mem,
      {numSubQuantizers_, queries.getSize(0), dimPerSubQuantizer_},
      stream);
    runTransposeAny(querySubQuantizerView, 0, 1, queriesTransposed, stream);

    DeviceTensor<float, 3, true> term3(
      mem,
      {numSubQuantizers_, queries.getSize(0), numSubQuantizerCodes_},
      stream);

    runIteratedMatrixMult(term3, false,
                          queriesTransposed, false,
                          pqCentroidsMiddleCode_, true,
                          -2.0f, 0.0f,
                          resources_->getBlasHandleCurrentDevice(),
                          stream);

    runTransposeAny(term3, 0, 1, term3Transposed, stream);

  }

  NoTypeTensor<3, true> term2;
  NoTypeTensor<3, true> term3;
#ifdef FAISS_USE_FLOAT16
  DeviceTensor<half, 3, true> term3Half;

  if (useFloat16LookupTables_) {
    printf("useFloat16LookupTables_\n");
    term3Half = toHalf(resources_, stream, term3Transposed);
    term2 = NoTypeTensor<3, true>(precomputedCodeHalf_);
    term3 = NoTypeTensor<3, true>(term3Half);
  }
#endif

  if (!useFloat16LookupTables_) {
   printf("useNO-Float16LookupTables_\n");
    term2 = NoTypeTensor<3, true>(precomputedCode_);
    term3 = NoTypeTensor<3, true>(term3Transposed);
  }
   // Compute ||y_R||^2 by treating
  // (sub q)(code id)(sub dim) as (sub q * code id)(sub dim)
  auto pqCentroidsMiddleCodeView =
    pqCentroidsMiddleCode_.view<2>(
      {numSubQuantizers_ * numSubQuantizerCodes_, dimPerSubQuantizer_});
  DeviceTensor<float, 1, true> subQuantizerNorms( mem,
    {numSubQuantizers_ * numSubQuantizerCodes_},stream);

  runL2Norm(pqCentroidsMiddleCodeView, subQuantizerNorms, true,
            stream);
   // outputVec("subQuantizerNorms",subQuantizerNorms.data(),10,stream);

  runPQScanMultiPassPrecomputedGraph(queries,
                                coarseDistances, // term 1
                                subQuantizerNorms,
                                term2, // term 2
                                term3, // term 3
                                coarseIndices,
                                graphDistances,
                                edgeInfo,
                                edgeDistInfo,
                                lambdaInfo,
                                useFloat16LookupTables_,
                                bytesPerVector_,
                                numSubQuantizers_,
                                numSubQuantizerCodes_,
                                deviceListDataPointers_,
                                deviceListIndexPointers_,
                                deviceListLambdaPointers_,
                                indicesOptions_,
                                deviceListLengths_,
                                maxListLength_,
                                k,
                                outDistances,
                                outIndices,
                                resources_);
}


void
IVFPQ::runPQPrecomputedCodes_(
  Tensor<float, 2, true>& queries,
  DeviceTensor<float, 2, true>& coarseDistances,
  DeviceTensor<int, 2, true>& coarseIndices,
  int k,
  Tensor<float, 2, true>& outDistances,
  Tensor<long, 2, true>& outIndices) {
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // Compute precomputed code term 3, - 2 * (x|y_R)
  // This is done via batch MM
  // {sub q} x {(query id)(sub dim) * (code id)(sub dim)'} =>
  // {sub q} x {(query id)(code id)}
  DeviceTensor<float, 3, true> term3Transposed(
    mem,
    {queries.getSize(0), numSubQuantizers_, numSubQuantizerCodes_},
    stream);

  // These allocations within are only temporary, so release them when
  // we're done to maximize free space
  {
    auto querySubQuantizerView = queries.view<3>(
      {queries.getSize(0), numSubQuantizers_, dimPerSubQuantizer_});
    DeviceTensor<float, 3, true> queriesTransposed(
      mem,
      {numSubQuantizers_, queries.getSize(0), dimPerSubQuantizer_},
      stream);
    runTransposeAny(querySubQuantizerView, 0, 1, queriesTransposed, stream);

    DeviceTensor<float, 3, true> term3(
      mem,
      {numSubQuantizers_, queries.getSize(0), numSubQuantizerCodes_},
      stream);

    runIteratedMatrixMult(term3, false,
                          queriesTransposed, false,
                          pqCentroidsMiddleCode_, true,
                          -2.0f, 0.0f,
                          resources_->getBlasHandleCurrentDevice(),
                          stream);

    runTransposeAny(term3, 0, 1, term3Transposed, stream);
  }

  NoTypeTensor<3, true> term2;
  NoTypeTensor<3, true> term3;
#ifdef FAISS_USE_FLOAT16
  DeviceTensor<half, 3, true> term3Half;

  if (useFloat16LookupTables_) {
    term3Half = toHalf(resources_, stream, term3Transposed);
    term2 = NoTypeTensor<3, true>(precomputedCodeHalf_);
    term3 = NoTypeTensor<3, true>(term3Half);
  }
#endif

  if (!useFloat16LookupTables_) {
    term2 = NoTypeTensor<3, true>(precomputedCode_);
    term3 = NoTypeTensor<3, true>(term3Transposed);
  }

  runPQScanMultiPassPrecomputed(queries,
                                coarseDistances, // term 1
                                term2, // term 2
                                term3, // term 3
                                coarseIndices,
                                useFloat16LookupTables_,
                                bytesPerVector_,
                                numSubQuantizers_,
                                numSubQuantizerCodes_,
                                deviceListDataPointers_,
                                deviceListIndexPointers_,
                                indicesOptions_,
                                deviceListLengths_,
                                maxListLength_,
                                k,
                                outDistances,
                                outIndices,
                                resources_);
}

void
IVFPQ::runPQNoPrecomputedCodes_(
  Tensor<float, 2, true>& queries,
  DeviceTensor<float, 2, true>& coarseDistances,
  DeviceTensor<int, 2, true>& coarseIndices,
  int k,
  Tensor<float, 2, true>& outDistances,
  Tensor<long, 2, true>& outIndices) {
  FAISS_ASSERT(!quantizer_->getUseFloat16());
  auto& coarseCentroids = quantizer_->getVectorsFloat32Ref();

  runPQScanMultiPassNoPrecomputed(queries,
                                  coarseCentroids,
                                  pqCentroidsInnermostCode_,
                                  coarseIndices,
                                  useFloat16LookupTables_,
                                  bytesPerVector_,
                                  numSubQuantizers_,
                                  numSubQuantizerCodes_,
                                  deviceListDataPointers_,
                                  deviceListIndexPointers_,
                                  indicesOptions_,
                                  deviceListLengths_,
                                  maxListLength_,
                                  k,
                                  outDistances,
                                  outIndices,
                                  resources_);
}

} } // namespace
