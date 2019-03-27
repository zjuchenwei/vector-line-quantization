/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "GpuIndexIVFPQ.h"
#include "../IndexFlat.h"
#include "../IndexIVFPQ.h"
#include "../filehelper.h"
#include "../ProductQuantizer.h"
#include "GpuIndexFlat.h"
#include "GpuResources.h"
#include "impl/IVFPQ.cuh"
#include "impl/FlatIndex.cuh"
#include "impl/Distance.cuh"
#include "impl/RemapIndices.h"
#include "impl/L2Norm.cuh"
#include "impl/InvertedListAppend.cuh"
#include "utils/CopyUtils.cuh"
#include "utils/DeviceUtils.h"
#include "utils/Tensor.cuh"
#include "utils/helper.cuh"
#include "utils/ConversionOperators.cuh"
#include "utils/DeviceTensor.cuh"
#include "utils/DeviceDefs.cuh"
#include "utils/Reductions.cuh"
#include "utils/PtxUtils.cuh"
#include "utils/HostTensor.cuh"
#include "utils/Transpose.cuh"
#include "utils/MemorySpace.h"
#include "utils/Select.cuh"
#include "utils/Limits.cuh"
#include <fstream>
#include <limits>
#include <unordered_map>
#include <cstring>
#include<unistd.h>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
namespace faiss { namespace gpu {



GpuIndexIVFPQ::GpuIndexIVFPQ(GpuResources* resources,
                             const faiss::IndexIVFPQ* index,
                             GpuIndexIVFPQConfig config) :
    GpuIndexIVF(resources,
                index->d,
                index->metric_type,
                index->nlist,
                config),
    ivfpqConfig_(config),
    subQuantizers_(0),
    bitsPerCode_(0),
    reserveMemoryVecs_(0),
    index_(nullptr) {
#ifndef FAISS_USE_FLOAT16
  FAISS_ASSERT(!ivfpqConfig_.useFloat16LookupTables);
#endif
  numedge_=32;
 edgeInfo_ = new int[index->nlist*numedge_];

 edgeDistInfo_ = new float[index->nlist*numedge_];
  nLambda_ = 256;
  lambdaInfo_ = new float[nLambda_];
   constInfo_ = new float[nLambda_];
   begin_=0;
   end_=nlist_;
  copyFrom(index);
}

GpuIndexIVFPQ::GpuIndexIVFPQ(GpuResources* resources,
                             int dims,
                             int nlist,
                             int subQuantizers,
                             int bitsPerCode,
                             faiss::MetricType metric,
                             GpuIndexIVFPQConfig config) :
    GpuIndexIVF(resources,
                dims,
                metric,
                nlist,
                config),
    ivfpqConfig_(config),
    subQuantizers_(subQuantizers),
    bitsPerCode_(bitsPerCode),
    reserveMemoryVecs_(0),
    index_(nullptr) {
#ifndef FAISS_USE_FLOAT16
  FAISS_ASSERT(!useFloat16LookupTables_);
#endif

  verifySettings_();

  // FIXME make IP work fully
  FAISS_ASSERT(this->metric_type == faiss::METRIC_L2);

  // We haven't trained ourselves, so don't construct the PQ index yet
  this->is_trained = false;
  numedge_=32;
  nLambda_ = 256;
  lambdaInfo_ = new float[nLambda_];
  constInfo_ = new float[nLambda_];
  edgeInfo_ = new int[nlist_*numedge_];
  begin_=0;
   end_=nlist_;
   w1_=1024;

  edgeDistInfo_ = new float[nlist_*numedge_];
}

GpuIndexIVFPQ::GpuIndexIVFPQ(GpuResources* resources,
                             int dims,
                             int nlist,
                             int subQuantizers,
                             int bitsPerCode,
                             int nedge,
                             int nLambda,
                             faiss::MetricType metric,
                             GpuIndexIVFPQConfig config) :
    GpuIndexIVF(resources,
                dims,
                metric,
                nlist,
                config),
    ivfpqConfig_(config),
    subQuantizers_(subQuantizers),
    bitsPerCode_(bitsPerCode),
    numedge_(nedge),
    nLambda_(nLambda),
    reserveMemoryVecs_(0),
    index_(nullptr) {
#ifndef FAISS_USE_FLOAT16
  FAISS_ASSERT(!useFloat16LookupTables_);
#endif

  verifySettings_();

  // FIXME make IP work fully
  FAISS_ASSERT(this->metric_type == faiss::METRIC_L2);

  // We haven't trained ourselves, so don't construct the PQ index yet
  this->is_trained = false;
  edgeInfo_ = new int[nlist_*numedge_];
  edgeDistInfo_ = new float[nlist_*numedge_];
  lambdaInfo_ = new float[nLambda_];
   constInfo_ = new float[nLambda_];
   begin_=0;
   end_=nlist_;
   w1_=1024;
}

GpuIndexIVFPQ::~GpuIndexIVFPQ() {
  delete index_;
  delete[] edgeInfo_;
  delete[] edgeDistInfo_;
  delete[] lambdaInfo_;
  delete[] constInfo_;
}


void
GpuIndexIVFPQ::copyFrom(const faiss::IndexIVFPQ* index) {
  DeviceScope scope(device_);

  // FIXME: support this
  FAISS_THROW_IF_NOT_MSG(index->metric_type == faiss::METRIC_L2,
                     "inner product unsupported");
  GpuIndexIVF::copyFrom(index);

  // Clear out our old data
  delete index_;
  index_ = nullptr;

  subQuantizers_ = index->pq.M;
  bitsPerCode_ = index->pq.nbits;

  // We only support this
  FAISS_ASSERT(index->pq.byte_per_idx == 1);
  FAISS_ASSERT(index->by_residual);
  FAISS_ASSERT(index->polysemous_ht == 0);
  ivfpqConfig_.usePrecomputedTables = (bool) index->use_precomputed_table;

  verifySettings_();

  // The other index might not be trained
  if (!index->is_trained) {
    return;
  }

  // Otherwise, we can populate ourselves from the other index
  this->is_trained = true;

  // Copy our lists as well
  // The product quantizer must have data in it
  FAISS_ASSERT(index->pq.centroids.size() > 0);
  index_ = new IVFPQ(resources_,
                     quantizer_->getGpuData(),
                     subQuantizers_,
                     bitsPerCode_,
                     (float*) index->pq.centroids.data(),
                     ivfpqConfig_.indicesOptions,
                     ivfpqConfig_.useFloat16LookupTables,
                     memorySpace_);
  // Doesn't make sense to reserve memory here
  index_->setPrecomputedCodes(ivfpqConfig_.usePrecomputedTables);

  // Copy database vectors, if any
  for (size_t i = 0; i < index->codes.size(); ++i) {
    auto& codes = index->codes[i];
    auto& ids = index->ids[i];

    FAISS_ASSERT(ids.size() * subQuantizers_ == codes.size());

    // GPU index can only support max int entries per list
    FAISS_THROW_IF_NOT_FMT(ids.size() <=
                       (size_t) std::numeric_limits<int>::max(),
                       "GPU inverted list can only support "
                       "%zu entries; %zu found",
                       (size_t) std::numeric_limits<int>::max(),
                       ids.size());

    index_->addCodeVectorsFromCpu(i, codes.data(), ids.data(), ids.size());
  }
}

void
GpuIndexIVFPQ::copyTo(faiss::IndexIVFPQ* index) const {
  DeviceScope scope(device_);

  // We must have the indices in order to copy to ourselves
  FAISS_THROW_IF_NOT_MSG(ivfpqConfig_.indicesOptions != INDICES_IVF,
                     "Cannot copy to CPU as GPU index doesn't retain "
                     "indices (INDICES_IVF)");

  GpuIndexIVF::copyTo(index);

  //
  // IndexIVFPQ information
  //
  index->by_residual = true;
  index->use_precomputed_table = 0;
  index->code_size = subQuantizers_;
  index->pq = faiss::ProductQuantizer(this->d, subQuantizers_, bitsPerCode_);

  index->do_polysemous_training = false;
  index->polysemous_training = nullptr;

  index->scan_table_threshold = 0;
  index->max_codes = 0;
  index->polysemous_ht = 0;
  index->codes.clear();
  index->codes.resize(nlist_);
  index->precomputed_table.clear();

  if (index_) {
    // Copy the inverted lists
    for (int i = 0; i < nlist_; ++i) {
      index->ids[i] = getListIndices(i);
      index->codes[i] = getListCodes(i);
    }

    // Copy PQ centroids
    auto devPQCentroids = index_->getPQCentroids();
    index->pq.centroids.resize(devPQCentroids.numElements());

    fromDevice<float, 3>(devPQCentroids,
                         index->pq.centroids.data(),
                         resources_->getDefaultStream(device_));

    if (ivfpqConfig_.usePrecomputedTables) {
      index->precompute_table();
    }
  }
}

void
GpuIndexIVFPQ::reserveMemory(size_t numVecs) {
  reserveMemoryVecs_ = numVecs;
  if (index_) {
    DeviceScope scope(device_);
    index_->reserveMemory(numVecs);
  }
}

void
GpuIndexIVFPQ::setPrecomputedCodes(bool enable) {
  ivfpqConfig_.usePrecomputedTables = enable;
  if (index_) {
    DeviceScope scope(device_);
    index_->setPrecomputedCodes(enable);
  }

  verifySettings_();
}

bool
GpuIndexIVFPQ::getPrecomputedCodes() const {
  return ivfpqConfig_.usePrecomputedTables;
}

int
GpuIndexIVFPQ::getNumSubQuantizers() const {
  return subQuantizers_;
}

int
GpuIndexIVFPQ::getBitsPerCode() const {
  return bitsPerCode_;
}

int
GpuIndexIVFPQ::getCentroidsPerSubQuantizer() const {
  return utils::pow2(bitsPerCode_);
}

size_t
GpuIndexIVFPQ::reclaimMemory() {
  if (index_) {
    DeviceScope scope(device_);
    return index_->reclaimMemory();
  }

  return 0;
}

void
GpuIndexIVFPQ::reset() {
  if (index_) {
    DeviceScope scope(device_);

    index_->reset();
    this->ntotal = 0;
  } else {
    FAISS_ASSERT(this->ntotal == 0);
  }
}

void
GpuIndexIVFPQ::trainResidualQuantizer_(Index::idx_t n, const float* x) {
  // Code largely copied from faiss::IndexIVFPQ
  // FIXME: GPUize more of this
  n = std::min(n, (Index::idx_t) (1 << bitsPerCode_) * 128);

  if (this->verbose) {
    printf("computing residuals\n");
  }

  std::vector<int> assign(n);
  quantizer_->assignFlat (n, x, assign.data());

 std::vector<int> assign1(n);
  std::vector<float> lambdaf(n);
 std::vector<uint8_t> lambda(n);

  quantizer_->assign1 (n,d,x, assign.data(),assign1.data(),lambdaf.data(),edgeInfo_,edgeDistInfo_,nlist_,numedge_);
  std::vector<float> residuals(n * d);

    Clustering clusLambda(1, nLambda_,cp_);
    clusLambda.verbose = verbose;
    GpuIndexFlatConfig config = ivfConfig_.flatConfig;
    config.device = device_;
    GpuIndexFlat* quantizer = new GpuIndexFlatL2(resources_, 1, config);

    clusLambda.train (n, lambdaf.data(), *quantizer);
    memcpy(lambdaInfo_, clusLambda.centroids.data(), sizeof(*lambdaInfo_) * 1 * nLambda_);

    quantizer_->assignLambda(n,lambdaf.data(),lambda.data(), lambdaInfo_, nLambda_);

  quantizer_->compute_residual(n,x,residuals.data(),edgeInfo_,lambda.data(),lambdaInfo_,numedge_,nlist_,assign1.data());
  if (this->verbose) {
    printf("training %d x %d product quantizer on %ld vectors in %dD\n",
           subQuantizers_, getCentroidsPerSubQuantizer(), n, this->d);
  }

  // Just use the CPU product quantizer to determine sub-centroids
  faiss::ProductQuantizer pq(this->d, subQuantizers_, bitsPerCode_);
  pq.verbose = this->verbose;
  pq.train(n, residuals.data());
   int pqCluster = 1 << bitsPerCode_;
  pqData_ = new float[pqCluster*d];
   memcpy (pqData_, pq.centroids.data(),pqCluster*d * sizeof (float));
  index_ = new IVFPQ(resources_,
                     quantizer_->getGpuData(),
                     numedge_,
                     subQuantizers_,
                     bitsPerCode_,
                     pqData_,
                     ivfpqConfig_.indicesOptions,
                     ivfpqConfig_.useFloat16LookupTables,
                     memorySpace_);
  if (reserveMemoryVecs_) {
    index_->reserveMemory(reserveMemoryVecs_);
  }

  index_->setPrecomputedCodes(ivfpqConfig_.usePrecomputedTables);
}

template <int RowTileSize>
__global__ void calConstQKernel( float* constq,
                Tensor<int, 2, true> edgeinfo,
                Tensor<float, 2, true> edgeDistinfo,
                 Tensor<int, 2, true> listIds2d,
                Tensor<int, 2, true> encodings,
                Tensor<float, 1, true> lambInfo,
                Tensor<uint8_t, 2, true> lambdaf,
                Tensor<half, 3, true> precomputedCode,
                Tensor<float, 1, true> subQuantizerNorms) {
    extern __shared__ char smemByte[];
    float* smem = (float*) smemByte;
    int numWarps = utils::divUp(blockDim.x, kWarpSize);
    int laneId = getLaneId();
    int warpId = threadIdx.x / kWarpSize;
    int nedge = edgeinfo.getSize(1);
    int nSubquantizerCodes = precomputedCode.getSize(2);
    int RowTileSize1= RowTileSize;
    bool lastRowTile = (blockIdx.x == (gridDim.x - 1));
    int rowStart = RowTileSize * blockIdx.x;
    float tmp=0;
    if (lastRowTile) {
        RowTileSize1 = encodings.getSize(0) - rowStart;
    }
    // We are handling the very end of the input matrix rows
    for (int row = 0; row < RowTileSize1; ++row) {

            float la = lambInfo[lambdaf[rowStart+row][0]];
            int listId = listIds2d[rowStart+row][0];
            int c = listId/nedge;
            int s =  edgeinfo.data()[listId];
            float edgeDist = edgeDistinfo.data()[listId];
            float dist = (la*la-la)*edgeDist;
            int code  = encodings[rowStart+row][threadIdx.x];
            float term2 =ConvertTo<float>::to(precomputedCode[s][threadIdx.x][code]);
            term2 -= ConvertTo<float>::to(precomputedCode[c][threadIdx.x][code]);
            //tmp= la*term2+ subQuantizerNorms[threadIdx.x*nSubquantizerCodes+code] ;
            tmp = la*term2;
             __syncthreads();
            tmp=warpReduceAllSum(tmp);

            if (laneId == 0) {
                smem[warpId] = tmp;
            }

            // Sum across warps
            if (warpId == 0) {
                tmp = laneId < numWarps ?
                              smem[laneId] : 0;
                tmp= warpReduceAllSum(tmp);
                 if (laneId == 0) {
                    constq[rowStart+row] = tmp;
                    constq[rowStart+row] +=dist;
                }

            }
            __syncthreads();

    }

}

template <int RowTileSize>
__global__ void calConstQKernel1( float* constq,
                Tensor<int, 2, true> edgeinfo,
                Tensor<float, 2, true> edgeDistinfo,
                 Tensor<int, 2, true> listIds2d,
                Tensor<int, 2, true> encodings,
                Tensor<float, 1, true> lambInfo,
                Tensor<float, 2, true> lambda,
                Tensor<half, 3, true> precomputedCode,
                Tensor<float, 1, true> subQuantizerNorms) {
    extern __shared__ char smemByte[];
    float* smem = (float*) smemByte;
    int numWarps = utils::divUp(blockDim.x, kWarpSize);
    int nedge = edgeinfo.getSize(1);
    int nSubquantizerCodes = precomputedCode.getSize(2);
    int RowTileSize1= RowTileSize;
    bool lastRowTile = (blockIdx.x == (gridDim.x - 1));
    int rowStart = RowTileSize * blockIdx.x;
    float tmp=0;
    if (lastRowTile) {
        RowTileSize1 = encodings.getSize(0) - rowStart;
    }
    // We are handling the very end of the input matrix rows
    for (int row = 0; row < RowTileSize1; ++row) {
         if(threadIdx.x==0){
            for(int i=0 ; i< encodings.getSize(1);i++){

                float la = lambda[rowStart+row][0];
                int listId = listIds2d[rowStart+row][0];
                int c = listId/nedge;
                int s =  edgeinfo.data()[listId];
                float edgeDist = edgeDistinfo.data()[listId];
                // float dist = (la*la-la)*edgeDist;
                int code  = encodings[rowStart+row][i];
                float term2 =(1-la)*ConvertTo<float>::to(precomputedCode[c][i][code]);
                term2 += la*ConvertTo<float>::to(precomputedCode[s][i][code]);
                tmp+= term2+ subQuantizerNorms[i*nSubquantizerCodes+code] ;

            }
            constq[rowStart+row] = tmp;

         }


    }

}

 void GpuIndexIVFPQ::calConstQ (float* constq,
                Tensor<int, 2, true>& edgeinfo,
                Tensor<float, 2, true>& edgeDistinfo,
                 Tensor<int, 2, true>& listIds2d,
                Tensor<int, 2, true>& encodings,
                Tensor<float, 1, true> lambInfo,
                Tensor<uint8_t, 2, true> lambda,
                Tensor<half, 3, true>& precomputedCode,
                Tensor<float, 1, true>& subQuantizerNorms){

       auto& mem = resources_->getMemoryManagerCurrentDevice();
       auto stream = resources_->getDefaultStream(device_);

       int maxThreads = getMaxThreadsCurrentDevice();
      auto devconstq = toDevice<float, 2>(resources_,
                                         device_,
                                         constq,
                                         stream,
                                         {(int) encodings.getSize(0), (int) 1});
        constexpr int rowTileSize = 8;
        int numThreads = min(encodings.getSize(1), maxThreads);
        auto grid = dim3(utils::divUp(encodings.getSize(0), rowTileSize));
        auto block = dim3(numThreads);
        auto smem = sizeof(float) * utils::divUp(numThreads, kWarpSize);
        calConstQKernel<rowTileSize><<<grid,block,smem,stream>>>(devconstq.data(),
                                                        edgeinfo,edgeDistinfo,listIds2d,
                                                        encodings,lambInfo,lambda,precomputedCode,
                                                         subQuantizerNorms);
        outputVec("devconstq",devconstq.data(),10,stream);

      fromDevice<float, 2>(devconstq, constq, stream);

  }

  void calConstQBase (float* constq,
                Tensor<int, 2, true>& edgeinfo,
                Tensor<float, 2, true>& edgeDistinfo,
                 Tensor<int, 2, true>& listIds2d,
                Tensor<int, 2, true>& encodings,
                 Tensor<float, 1, true> lambInfo,
               Tensor<uint8_t, 2, true> lambda,
                Tensor<half, 3, true>& precomputedCode,
                Tensor<float, 1, true>& subQuantizerNorms,cudaStream_t stream){


       int maxThreads = getMaxThreadsCurrentDevice();
        constexpr int rowTileSize = 8;
        int numThreads = min(encodings.getSize(1), maxThreads);
        auto grid = dim3(utils::divUp(encodings.getSize(0), rowTileSize));
        auto block = dim3(numThreads);
        auto smem = sizeof(float) * utils::divUp(numThreads, kWarpSize);
        calConstQKernel<rowTileSize><<<grid,block,smem,stream>>>(constq,
                                                        edgeinfo,edgeDistinfo,listIds2d,
                                                        encodings,lambInfo,lambda,precomputedCode,
                                                         subQuantizerNorms);


  }




int GpuIndexIVFPQ::classifyAndAddVectors(Tensor<float, 2, true>& vecs,
                             Tensor<long, 1, true>& indices) {
  FAISS_ASSERT(vecs.getSize(0) == indices.getSize(0));
  FAISS_ASSERT(vecs.getSize(1) == index_->dim_);

  //FAISS_ASSERT(!index_->quantizer_->getUseFloat16());
  //auto& coarseCentroids = index_->quantizer_->getVectorsFloat32Ref();
  auto& mem = index_->resources_->getMemoryManagerCurrentDevice();
  auto stream = index_->resources_->getDefaultStreamCurrentDevice();

  // Number of valid vectors that we actually add; we return this
  int numAdded = 0;

  // We don't actually need this
  DeviceTensor<float, 2, true> listDistance(mem, {vecs.getSize(0), 1}, stream);
  // We use this
  DeviceTensor<int, 2, true> listIds2d(mem, {vecs.getSize(0), 1}, stream);
  auto listIds = listIds2d.view<1>({vecs.getSize(0)});

   DeviceTensor<int, 2, true> listIds2d1(mem, {vecs.getSize(0), 1}, stream);
     DeviceTensor<float, 2, true> lambdaf(mem, {vecs.getSize(0), 1}, stream);
   DeviceTensor<uint8_t, 2, true> lambda(mem, {vecs.getSize(0), 1}, stream);
  auto listIds1 = listIds2d1.view<1>({vecs.getSize(0)});

   quantizer_->getGpuData()->query(vecs, 1, listDistance, listIds2d, true);

     //outputVecInt("listIds2d",listIds2d.data(),100, stream);
    //outputVec("listDistance",listDistance.data(),10, stream);

  auto edgedistinfoV = toDevice<float, 2>(resources_,
                                         device_,
                                         edgeDistInfo_,
                                         stream,
                                         {(int) nlist_, (int) numedge_});

  // Convert and copy int indices out
  auto edgeinfoV = toDevice<int, 2>(resources_,
                                                     device_,
                                                     edgeInfo_,
                                                     stream,
                                                     {(int) nlist_, (int) numedge_});
  quantizer_->assign1Base (vecs, listIds2d,listIds2d1,lambdaf,edgeinfoV,edgedistinfoV);
  //outputVecInt("listIds2d1",listIds2d1.data(),100, stream);
  //outputVecUint8("lambda",lambda.data(),100, stream);
  DeviceTensor<float, 2, true> residuals(mem, {vecs.getSize(0), vecs.getSize(1)}, stream);
  quantizer_->assignLambda(vecs.getSize(0),lambdaf.data(),lambda.data(), lambdaInfo_, nLambda_);
   HostTensor<uint8_t, 2, true>lambdaHost(lambda, stream);
  quantizer_->compute_residual(vecs.getSize(0),vecs.data(),residuals.data(),edgeInfo_,lambdaHost.data(),lambdaInfo_,numedge_,nlist_,listIds2d1.data());
  // outputVec("residuals",residuals.data(),100, stream);
// outputVecInt("listIds1",listIds1.data(),10, stream);
  // Copy the lists that we wish to append to back to the CPU
  // FIXME: really this can be into pinned memory and a true async
  // copy on a different stream; we can start the copy early, but it's
  // tiny
  HostTensor<int, 1, true> listIdsHost(listIds1, stream);

  // Calculate the residual for each closest centroid
 // DeviceTensor<float, 2, true> residuals(
  //  mem, {vecs.getSize(0), vecs.getSize(1)}, stream);

 // runCalcResidual(vecs, coarseCentroids, listIds, residuals, stream);

  // Residuals are in the form
  // (vec x numSubQuantizer x dimPerSubQuantizer)
  // transpose to
  // (numSubQuantizer x vec x dimPerSubQuantizer)

  printf("subQuantizers_:%d,residuals.getSize(0):%d\n",subQuantizers_,residuals.getSize(0));
  int dimPerSubQuantizer_ =  vecs.getSize(1)/subQuantizers_;
  auto residualsView = residuals.view<3>(
    {residuals.getSize(0), subQuantizers_, dimPerSubQuantizer_});

  DeviceTensor<float, 3, true> residualsTranspose(
    mem,
    {subQuantizers_, residuals.getSize(0), dimPerSubQuantizer_},
    stream);

  runTransposeAny(residualsView, 0, 1, residualsTranspose, stream);
   //outputVec("residualsTranspose",residualsTranspose.data(),10, stream);
  // Get the product quantizer centroids in the form
  // (numSubQuantizer x numSubQuantizerCodes x dimPerSubQuantizer)
  // which is pqCentroidsMiddleCode_

  // We now have a batch operation to find the top-1 distances:
  // batch size: numSubQuantizer
  // centroids: (numSubQuantizerCodes x dimPerSubQuantizer)
  // residuals: (vec x dimPerSubQuantizer)
  // => (numSubQuantizer x vec x 1)

  DeviceTensor<float, 3, true> closestSubQDistance(
    mem, {subQuantizers_, residuals.getSize(0), 1}, stream);
  DeviceTensor<int, 3, true> closestSubQIndex(
    mem, {subQuantizers_, residuals.getSize(0), 1}, stream);

  for (int subQ = 0; subQ < subQuantizers_; ++subQ) {
    auto closestSubQDistanceView = closestSubQDistance[subQ].view();
    auto closestSubQIndexView = closestSubQIndex[subQ].view();

    auto pqCentroidsMiddleCodeView = index_->pqCentroidsMiddleCode_[subQ].view();
    //  outputVec("pqCentroidsMiddleCodeView",pqCentroidsMiddleCodeView.data(),10, stream);
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
    {subQuantizers_, residuals.getSize(0)});

  // Transpose this for easy use
  DeviceTensor<int, 2, true> encodings(
    mem, {residuals.getSize(0), subQuantizers_}, stream);

  runTransposeAny(closestSubQIndexView, 0, 1, encodings, stream);
   int numSubQuantizerCodes = index_->numSubQuantizerCodes_;
   auto pqCentroidsMiddleCodeView =
    index_->pqCentroidsMiddleCode_.view<2>(
      {subQuantizers_ * numSubQuantizerCodes, dimPerSubQuantizer_});
  DeviceTensor<float, 1, true> subQuantizerNorms( mem,
    {subQuantizers_   * numSubQuantizerCodes},stream);

  runL2Norm(pqCentroidsMiddleCodeView, subQuantizerNorms, true,
            stream);

  // Transpose this for easy use
 // DeviceTensor<float, 2, true> constqf(
 //   mem, {residuals.getSize(0),1}, stream);
  DeviceTensor<uint8_t, 2, true> constq(
    mem, {residuals.getSize(0),1}, stream);
 //  auto devlambdaInfo = toDevice<float, 1>(resources_,
  //                                       device_,
   //                                      lambdaInfo_,
   //                                      stream,
    //                                     {(int) nLambda_});

  //calConstQBase(constqf.data(),edgeinfoV,edgedistinfoV,listIds2d1,
   //          encodings,devlambdaInfo,lambda,index_->precomputedCodeHalf_,subQuantizerNorms,stream);

 //quantizer_->assignLambda(vecs.getSize(0),constqf.data(),constq.data(), constInfo_, nLambda_);

 // outputVec("constqf",constqf.data(),10,stream);
  //  outputVecUint8("constq",constq.data(),10,stream);
 // outputVecInt("encodings",encodings.data(),100, stream);
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
    // Add vector could be invalid (contains NaNs etc)
    if (listId < 0) {
      listOffsetHost[i] = -1;
      continue;
    }

    FAISS_ASSERT(listId < index_->numLists_);
    ++numAdded;

    int offset = index_->listOffsetToUserCode_[listId].size() / index_->bytesPerVector_;

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
    // auto& codes = index_->deviceListData_[counts.first];
      //printf("counts.first:%d,counts.second:%d\n",counts.first,counts.second);
    //  codes->resize(codes->size() + counts.second * index_->bytesPerVector_,
                   //stream);
    // int newNumVecs = (int) (codes->size() / index_->bytesPerVector_);

       auto& userCodes = index_->listOffsetToUserCode_[counts.first];
      userCodes.resize(userCodes.size() + counts.second * index_->bytesPerVector_);
      int newNumVecs = (int) (userCodes.size() / index_->bytesPerVector_);

      auto& indices = index_->deviceListIndices_[counts.first];
      if ((index_->indicesOptions_ == INDICES_32_BIT) ||
          (index_->indicesOptions_ == INDICES_64_BIT)) {
        size_t indexSize =
          (index_->indicesOptions_ == INDICES_32_BIT) ? sizeof(int) : sizeof(long);

        indices->resize(indices->size() + counts.second * indexSize, stream);
      } else if (index_->indicesOptions_ == INDICES_CPU) {
        // indices are stored on the CPU side
        FAISS_ASSERT(counts.first < index_->listOffsetToUserIndex_.size());

        auto& userIndices = index_->listOffsetToUserIndex_[counts.first];
        userIndices.resize(newNumVecs);
      } else {
        // indices are not stored on the GPU or CPU side
        FAISS_ASSERT(index_->indicesOptions_ == INDICES_IVF);
      }
      size_t lambdaSize = sizeof(uint8_t);
      auto& lambdas = index_->deviceListLambdas_[counts.first];
       lambdas->resize(lambdas->size() + counts.second * lambdaSize, stream);

    // size_t constSize = sizeof(uint8_t);
     // auto& consts = index_->deviceListConsts_[counts.first];
      // consts->resize(lambdas->size() + counts.second * lambdaSize, stream);


      // This is used by the multi-pass query to decide how much scratch
      // space to allocate for intermediate results
      index_->maxListLength_ = std::max(index_->maxListLength_, newNumVecs);
    }

    // Update all pointers and sizes on the device for lists that we
    // appended to
    {
      std::vector<int> listIds(assignCounts.size());
      int i = 0;
      for (auto& counts : assignCounts) {
        listIds[i++] = counts.first;
      }

      index_->updateDeviceListInfo_(listIds, stream);
    }
  }

    HostTensor<int, 2, true> hostCodes(encodings, stream);

   for (int i = 0; i < hostCodes.getSize(0); ++i) {
          int listId = listIdsHost[i];

      // Add vector could be invalid (contains NaNs etc)
      if (listId < 0) {
        continue;
      }

      int offset = listOffsetHost[i];

      FAISS_ASSERT(listId < index_->listOffsetToUserCode_.size());
      auto& userCodes = index_->listOffsetToUserCode_[listId];

      int newNumVecs = (int) (userCodes.size() / index_->bytesPerVector_);
      FAISS_ASSERT(offset < newNumVecs);
       for(int j =0;j<index_->bytesPerVector_;j++)
      userCodes[offset*index_->bytesPerVector_+j] = (uint8_t)hostCodes[i][j];
  }

  // If we're maintaining the indices on the CPU side, update our
  // map. We already resized our map above.
  if (index_->indicesOptions_ == INDICES_CPU) {
    // We need to maintain the indices on the CPU side
    HostTensor<long, 1, true> hostIndices(indices, stream);

    for (int i = 0; i < hostIndices.getSize(0); ++i) {
      int listId = listIdsHost[i];

      // Add vector could be invalid (contains NaNs etc)
      if (listId < 0) {
        continue;
      }

      int offset = listOffsetHost[i];

      FAISS_ASSERT(listId < index_->listOffsetToUserIndex_.size());
      auto& userIndices = index_->listOffsetToUserIndex_[listId];

      FAISS_ASSERT(offset < userIndices.size());
      userIndices[offset] = hostIndices[i];
    }
  }

  // We similarly need to actually append the new encoded vectors
  {
    DeviceTensor<int, 1, true> listOffset(mem, listOffsetHost, stream);
        //outputVecInt("listOffset",listOffset.data(),10, stream);
    // This kernel will handle appending each encoded vector + index to
    // the appropriate list
    runIVFPQInvertedListAppend(listIds1,
                               listOffset,
                               encodings,
                               indices,
                               lambda,
                               constq,
                               index_->deviceListDataPointers_,
                               index_->deviceListIndexPointers_,
                               index_->deviceListLambdaPointers_,
                               index_->deviceListConstPointers_,
                               index_->indicesOptions_,
                               stream);


       //outputVecInt("listOffset111",listOffset.data(),15, stream);
  }

  return numAdded;
}


void GpuIndexIVFPQ::classifyAndAddVectors1(Tensor<float, 2, true>& vecs,
                             Tensor<long, 1, true>& indices, int* assign1) {
  FAISS_ASSERT(vecs.getSize(0) == indices.getSize(0));
  FAISS_ASSERT(vecs.getSize(1) == index_->dim_);
  //FAISS_ASSERT(!index_->quantizer_->getUseFloat16());
  //auto& coarseCentroids = index_->quantizer_->getVectorsFloat32Ref();
  auto& mem = index_->resources_->getMemoryManagerCurrentDevice();
  auto stream = index_->resources_->getDefaultStreamCurrentDevice();

  // Number of valid vectors that we actually add; we return this

  // We don't actually need this
  DeviceTensor<float, 2, true> listDistance(mem, {vecs.getSize(0), 1}, stream);
  // We use this
  DeviceTensor<int, 2, true> listIds2d(mem, {vecs.getSize(0), 1}, stream);
  auto listIds = listIds2d.view<1>({vecs.getSize(0)});

   quantizer_->getGpuData()->query(vecs, 1, listDistance, listIds2d, true);
   outputVecInt("listIds2d",listIds2d.data(),15, stream);
   HostTensor<int, 2, true> hostListIds2d(listIds2d, stream);

   memcpy((char*)assign1,(char*)hostListIds2d.data(),sizeof(int)*listIds2d.getSize(0));
    ntotal +=vecs.getSize(0);

}

void
GpuIndexIVFPQ::buildGraph_(faiss::Index::idx_t n, const float* x) {
  if (quantizer_->ntotal == 0) {
    // nothing to do
    return;
  }



  DeviceScope scope(device_);

  quantizer_->buildGraph(quantizer_->ntotal,numedge_, edgeDistInfo_, edgeInfo_);
  //for(int i=0;i<numedge_;i++){
   // printf("assign %d : %d , dis %d : %f\n",i,edgeInfo_[i],i,edgeDistInfo_[i]);
 // }
  FAISS_ASSERT(quantizer_->ntotal == nlist_);
}



void GpuIndexIVFPQ::readTreeFromFile(Index::idx_t n, const float* x,const std::string& _name) {

    std::ifstream f(_name.c_str(), std::ofstream::in | std::ofstream::binary);
    uint param;
	f >> param;
	f >> param;
	f >> param;
	f >> param;
	f >> param;
	f >> param;

	f.ignore(1);


	float * cb1Host = new float[nlist_ * d];
	f.read((char*) cb1Host, nlist_ * d * sizeof(float));
	quantizer_->reset();
    quantizer_->add(nlist_, cb1Host);
    quantizer_->is_trained = true;
    int pqCluster = 1 << bitsPerCode_;
    int dimPerSubQuantizer_ = d/subQuantizers_;
	faiss::ProductQuantizer pq(this->d, subQuantizers_, bitsPerCode_);
    pq.verbose = this->verbose;
    f.read((char*) pq.centroids.data(),   pqCluster * d * sizeof(float));
	f.read((char*) edgeInfo_, numedge_*nlist_ * sizeof(int));
	f.read((char*) edgeDistInfo_, numedge_*nlist_ * sizeof(float));
     std::vector<int> assign(n);
  quantizer_->assignFlat (n, x, assign.data());

   std::vector<int> assign1(n);
   std::vector<float> lambdaf(n);
   std::vector<uint8_t> lambda(n);

    quantizer_->assign1 (n,d,x, assign.data(),assign1.data(),lambdaf.data(),edgeInfo_,edgeDistInfo_,nlist_,numedge_);

    Clustering clusLambda(1, nLambda_,cp_);
    clusLambda.verbose = verbose;
    GpuIndexFlatConfig config = ivfConfig_.flatConfig;
    config.device = device_;
    GpuIndexFlat* quantizer = new GpuIndexFlatL2(resources_, 1, config);

    clusLambda.train (n, lambdaf.data(), *quantizer);
    memcpy(lambdaInfo_, clusLambda.centroids.data(), sizeof(*lambdaInfo_) * 1 * nLambda_);

    index_ = new IVFPQ(resources_,
                     quantizer_->getGpuData(),
                     numedge_,
                     subQuantizers_,
                     bitsPerCode_,
                     pq.centroids.data(),
                     true,
                     ivfpqConfig_.indicesOptions,
                     ivfpqConfig_.useFloat16LookupTables,
                     memorySpace_);
  if (reserveMemoryVecs_) {
    index_->reserveMemory(reserveMemoryVecs_);


  }

  index_->setPrecomputedCodes(ivfpqConfig_.usePrecomputedTables);
     this->is_trained = true;

  /* auto& mem = resources_->getMemoryManagerCurrentDevice();
    auto stream =resources_->getDefaultStreamCurrentDevice();

    auto vecs = toDevice<float, 2>(resources_,
                                         device_,
                                          const_cast<float*>(x),
                                         stream,
                                         {(int) n, (int) d});

  auto edgedistinfoV = toDevice<float, 2>(resources_,
                                         device_,
                                         edgeDistInfo_,
                                         stream,
                                         {(int) nlist_, (int) numedge_});

  // Convert and copy int indices out
    auto edgeinfoV = toDevice<int, 2>(resources_,
                                                     device_,
                                                     edgeInfo_,
                                                     stream,
                                                     {(int) nlist_, (int) numedge_});
     auto listIds2d1 = toDevice<int, 2>(resources_,
                                                     device_,
                                                     assign1.data(),
                                                     stream,
                                                     {(int) n, (int) 1});
     auto devlambdaf = toDevice<float, 2>(resources_,
                                         device_,
                                         lambdaf.data(),
                                         stream,
                                         {(int) n, (int) 1});



  DeviceTensor<float, 2, true> residuals(mem, {vecs.getSize(0), vecs.getSize(1)}, stream);
   std::vector<float> constq(n);

  quantizer_->compute_residual(vecs.getSize(0),vecs.data(),residuals.data(),edgeInfo_,lambdaf.data(),numedge_,nlist_,listIds2d1.data());


  auto residualsView = residuals.view<3>(
    {residuals.getSize(0), subQuantizers_, dimPerSubQuantizer_});

  DeviceTensor<float, 3, true> residualsTranspose(
    mem,
    {subQuantizers_, residuals.getSize(0), dimPerSubQuantizer_},
    stream);

  runTransposeAny(residualsView, 0, 1, residualsTranspose, stream);
   //outputVec("residualsTranspose",residualsTranspose.data(),10, stream);
  // Get the product quantizer centroids in the form
  // (numSubQuantizer x numSubQuantizerCodes x dimPerSubQuantizer)
  // which is pqCentroidsMiddleCode_

  // We now have a batch operation to find the top-1 distances:
  // batch size: numSubQuantizer
  // centroids: (numSubQuantizerCodes x dimPerSubQuantizer)
  // residuals: (vec x dimPerSubQuantizer)
  // => (numSubQuantizer x vec x 1)

  DeviceTensor<float, 3, true> closestSubQDistance(
    mem, {subQuantizers_, residuals.getSize(0), 1}, stream);
  DeviceTensor<int, 3, true> closestSubQIndex(
    mem, {subQuantizers_, residuals.getSize(0), 1}, stream);

  for (int subQ = 0; subQ < subQuantizers_; ++subQ) {
    auto closestSubQDistanceView = closestSubQDistance[subQ].view();
    auto closestSubQIndexView = closestSubQIndex[subQ].view();

    auto pqCentroidsMiddleCodeView = index_->pqCentroidsMiddleCode_[subQ].view();
    //  outputVec("pqCentroidsMiddleCodeView",pqCentroidsMiddleCodeView.data(),10, stream);
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
    {subQuantizers_, residuals.getSize(0)});

  // Transpose this for easy use
  DeviceTensor<int, 2, true> encodings(
    mem, {residuals.getSize(0), subQuantizers_}, stream);

  runTransposeAny(closestSubQIndexView, 0, 1, encodings, stream);
  outputVecInt("encodings",encodings.data(),10,stream);
  auto pqCentroidsMiddleCodeView =
    index_->pqCentroidsMiddleCode_.view<2>(
      {subQuantizers_ * pqCluster, dimPerSubQuantizer_});
  DeviceTensor<float, 1, true> subQuantizerNorms( mem,
    {subQuantizers_   * pqCluster},stream);

  runL2Norm(pqCentroidsMiddleCodeView, subQuantizerNorms, true,
            stream);
    outputVec("subQuantizerNorms",subQuantizerNorms.data(),10,stream);
      outputVec("edgedistinfoV",edgedistinfoV.data(),10,stream);
      outputVec("subQuantizerNorms",subQuantizerNorms.data(),10,stream);
      outputVecInt("listIds2d1",listIds2d1.data(),10,stream);
      //outputVec("precomputedCode_",index_->precomputedCode_.data(),10,stream);
  quantizer_->assignLambda(n,lambdaf.data(),lambda.data(), lambdaInfo_, nLambda_);

   auto devlambda = toDevice<uint8_t, 2>(resources_,
                                         device_,
                                         lambda.data(),
                                         stream,
                                         {(int) n, (int) 1});

    auto devlambdaInfo = toDevice<float, 1>(resources_,
                                         device_,
                                         lambdaInfo_,
                                         stream,
                                         {(int) nLambda_});

  calConstQ (constq.data(),edgeinfoV,edgedistinfoV,listIds2d1,
              encodings,devlambdaInfo,devlambda,index_->precomputedCodeHalf_,subQuantizerNorms);

   Clustering clusConst(1, nLambda_,cp_);
    clusConst.verbose = verbose;
    GpuIndexFlat* quantizer1 = new GpuIndexFlatL2(resources_, 1, config);
    clusConst.train (n, constq.data(), *quantizer1);
    memcpy(constInfo_, clusConst.centroids.data(), sizeof(*constInfo_) * 1 * nLambda_);*/

}

void
GpuIndexIVFPQ::train(Index::idx_t n, const float* x) {
  DeviceScope scope(device_);

  if (this->is_trained) {
    FAISS_ASSERT(quantizer_->is_trained);
    FAISS_ASSERT(quantizer_->ntotal == nlist_);
    FAISS_ASSERT(index_);
    return;
  }

  FAISS_ASSERT(!index_);

  trainQuantizer_(n, x);
  buildGraph_(n,x);
  trainResidualQuantizer_(n, x);

  this->is_trained = true;
}


void
GpuIndexIVFPQ::addImpl_(Index::idx_t n,
                        const float* x,
                        const Index::idx_t* xids) {
  // Device is already set in GpuIndex::addInternal_
  FAISS_ASSERT(index_);
  FAISS_ASSERT(n > 0);

  auto stream = resources_->getDefaultStreamCurrentDevice();

  auto deviceVecs =
    toDevice<float, 2>(resources_,
                       device_,
                       const_cast<float*>(x),
                       stream,
                       {(int) n, index_->getDim()});

  auto deviceIndices =
    toDevice<Index::idx_t, 1>(resources_,
                              device_,
                              const_cast<Index::idx_t*>(xids),
                              stream,
                              {(int) n});

  ntotal += classifyAndAddVectors(deviceVecs, deviceIndices);
    std::cout <<"ListLength"<<getListLength(135*numedge_+1)<<std::endl;
  //	for (int i = 0; i <getListLength(135*numedge_+1); i++){
  	//  std::cout << "\t" << getListIndices(135*numedge_+1)[i];
  	//  std::cout << "\t" <<(int)getListLambdas(135*numedge_+1)[i];
  //	}

	std::cout << std::endl;
	std::cout << std::endl;
}


void
GpuIndexIVFPQ::addImpl1_(Index::idx_t n,
                        const float* x,
                        const Index::idx_t* xids, int* assign1) {
  // Device is already set in GpuIndex::addInternal_
  FAISS_ASSERT(index_);
  FAISS_ASSERT(n > 0);

  auto stream = resources_->getDefaultStreamCurrentDevice();

  auto deviceVecs =
    toDevice<float, 2>(resources_,
                       device_,
                       const_cast<float*>(x),
                       stream,
                       {(int) n, index_->getDim()});

  auto deviceIndices =
    toDevice<Index::idx_t, 1>(resources_,
                              device_,
                              const_cast<Index::idx_t*>(xids),
                              stream,
                              {(int) n});
        printf("classifyAndAddVectors1\n");
   classifyAndAddVectors1(deviceVecs, deviceIndices,assign1);
}



int* GpuIndexIVFPQ::add_with_ids1(Index::idx_t n,
                       const float* x,
                       const Index::idx_t* ids) {
  DeviceScope scope(device_);
  int* assign1 = new int[n] ;
  FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");
  // Default size for which we page add or search
constexpr size_t kAddPageSize = (size_t) 256 * 1024 * 1024;
// Or, maximum number of vectors to consider per page of add or search
constexpr size_t kAddVecSize = (size_t) 512 * 1024;

  if (n > 0) {
    size_t totalSize = n * (size_t) this->d * sizeof(float);

    if (totalSize > kAddPageSize || n > kAddVecSize) {
      // How many vectors fit into kAddPageSize?
      size_t maxNumVecsForPageSize =
        kAddPageSize / ((size_t) this->d * sizeof(float));

      // Always add at least 1 vector, if we have huge vectors
      maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t) 1);

      size_t tileSize = std::min((size_t) n, maxNumVecsForPageSize);
      tileSize = std::min(tileSize, kAddVecSize);

      for (size_t i = 0; i < n; i += tileSize) {
        size_t curNum = std::min(tileSize, n - i);
        addImpl1_(curNum,
                 x + i * (size_t) this->d,
                 ids ? ids + i : nullptr,assign1+i);
      }
    } else {
      addImpl1_(n, x, ids,assign1);
    }
  }

  return assign1;
}

struct IntToLong {
  __device__ long operator()(int v) const { return (long) v; }
};

__global__ void mergekernel1(
                     Tensor<int, 2, true> intNns,
                      Tensor<faiss::Index::idx_t, 2, true>   nns,
                       Tensor<faiss::Index::idx_t, 1, true>  ids) {
      // Each block handles a single row of the distances (results)
      int row = blockIdx.x;

   // Whole warps must participate in the selection
      int limit = intNns.getSize(1);
      int rowNum = intNns.getSize(0);
      for(;row<rowNum;row+=gridDim.x){
          for (int i = threadIdx.x; i < limit; i += blockDim.x) {
            int rank= intNns[row][i];
            faiss::Index::idx_t id =ids[rank];
            nns[row][i] = id;
          }
      }



}


void GpuIndexIVFPQ::generateNNs(Tensor<float, 2, true>& vecs,Tensor<float, 2, true>& vecsq,
                             Tensor<long, 1, true>& indices, Tensor<long, 2, true>& nns,Tensor<float, 2, true>& dists) {
  FAISS_ASSERT(vecs.getSize(0) == indices.getSize(0));
  FAISS_ASSERT(vecs.getSize(1) == index_->dim_);
  //FAISS_ASSERT(!index_->quantizer_->getUseFloat16());
  //auto& coarseCentroids = index_->quantizer_->getVectorsFloat32Ref();
  auto& mem = index_->resources_->getMemoryManagerCurrentDevice();
  auto stream = index_->resources_->getDefaultStreamCurrentDevice();


   // FlatIndex only supports an interface returning int indices
  DeviceTensor<int, 2, true> outIntNns(
    mem,
    {(int) nns.getSize(0), (int) nns.getSize(1)}, stream);

    runL2Distance(resources_,
                  vecs,
                  nullptr, // no transposed storage
                  nullptr, // no precomputed norms
                  vecsq,
                  outIntNns.getSize(1),
                  dists,
                  outIntNns,
                  // We don't care about distances
                  true,
                  // Much larger tile size, since these vectors are a
                  // lot smaller than query vectors
                  1000);
     constexpr int kThreadsPerBlock = 128;

      auto block = dim3(kThreadsPerBlock);
      auto grid = dim3(256);
      //outputVecLong("indices",indices.data(),10, resources_->getDefaultStream(device_));

      mergekernel1<<<grid, block, 0, resources_->getDefaultStream(device_)>>>(outIntNns,nns,indices);
      //outputVecInt("outIntNns",outIntNns.data(),10, resources_->getDefaultStream(device_));
      //outputVecLong("nns",nns.data(),10, resources_->getDefaultStream(device_));


}


void GpuIndexIVFPQ::add_with_ids2(Index::idx_t n,Index::idx_t nq, uint kgt,
                       const float* x,
                       const float* xq,
                       const Index::idx_t* ids,Index::idx_t* nns,float* dists) {
  DeviceScope scope(device_);
  auto devX =
    toDevice<float, 2>(resources_,
                       device_,
                       const_cast<float*>(x),
                       resources_->getDefaultStream(device_),
                       {(int) n, (int) index_->getDim()});
   auto devXq =
    toDevice<float, 2>(resources_,
                       device_,
                       const_cast<float*>(xq),
                       resources_->getDefaultStream(device_),
                       {(int) nq, (int) index_->getDim()});
    auto devIds =
    toDevice<Index::idx_t, 1>(resources_,
                              device_,
                              const_cast<Index::idx_t*>(ids),
                            resources_->getDefaultStream(device_),
                              {(int) n});
 auto devDists =
    toDevice<float, 2>(resources_,
                       device_,
                       dists,
                       resources_->getDefaultStream(device_),
                       {(int) nq, (int) kgt});
 auto devLabels =
    toDevice<faiss::Index::idx_t, 2>(resources_,
                                     device_,
                                     nns,
                                     resources_->getDefaultStream(device_),
                                     {(int) nq, (int) kgt});

  generateNNs(devX,devXq,devIds, devLabels,devDists);

  fromDevice<float, 2>(
    devDists, dists, resources_->getDefaultStream(device_));
  fromDevice<faiss::Index::idx_t, 2>(
    devLabels, nns, resources_->getDefaultStream(device_));


}

void
GpuIndexIVFPQ::searchImpl_(faiss::Index::idx_t n,
                           const float* x,
                           faiss::Index::idx_t k,
                           float* distances,
                           faiss::Index::idx_t* labels) const {
  // Device is already set in GpuIndex::search
   printf("GpuIndexIVFPQ::searchImpl_\n");
  FAISS_ASSERT(index_);
  FAISS_ASSERT(n > 0);

  // Make sure arguments are on the device we desire; use temporary
  // memory allocations to move it if necessary
 auto devX =
    toDevice<float, 2>(resources_,
                       device_,
                       const_cast<float*>(x),
                       resources_->getDefaultStream(device_),
                       {(int) n, (int) index_->getDim()});
 auto devDistances =
    toDevice<float, 2>(resources_,
                       device_,
                       distances,
                       resources_->getDefaultStream(device_),
                       {(int) n, (int) k});
 auto devLabels =
    toDevice<faiss::Index::idx_t, 2>(resources_,
                                     device_,
                                     labels,
                                     resources_->getDefaultStream(device_),
                                     {(int) n, (int) k});
   auto devEdgeInfo =
    toDevice<int, 2>(resources_,
                       device_,
                       edgeInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nlist_, (int) numedge_});
    auto devEdgeDistInfo =
    toDevice<float, 2>(resources_,
                       device_,
                       edgeDistInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nlist_, (int) numedge_});
   auto devLambdaInfo =
    toDevice<float, 1>(resources_,
                       device_,
                       lambdaInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nLambda_});

    auto devConstInfo =
    toDevice<float, 1>(resources_,
                       device_,
                       constInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nLambda_});

 index_->queryGraph(devX,devEdgeInfo,devEdgeDistInfo,devLambdaInfo,devConstInfo,nprobe_, w1_,(int) k, begin_ , end_,devDistances,devLabels);
  //outputVec("devDistances",devDistances.data(),10, resources_->getDefaultStream(device_));
  // Copy back if necessary
  fromDevice<float, 2>(
    devDistances, distances, resources_->getDefaultStream(device_));
  fromDevice<faiss::Index::idx_t, 2>(
    devLabels, labels, resources_->getDefaultStream(device_));
}


template <int NumWarpQ, int NumThreadQ, int ThreadsPerBlock>
__global__ void mergekernel(
                     float* dists,
                    faiss::Index::idx_t* nns,
                     Tensor<float, 2, true> outDists,
                      Tensor<faiss::Index::idx_t, 2, true>   outLabels,
                      int k,
                      int n,int nprocess , float initK) {
      // Each block handles a single row of the distances (results)
      constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;
      __shared__ float smemK[kNumWarps * NumWarpQ];
      __shared__ int smemV[kNumWarps * NumWarpQ];

      int num =k*nprocess;
      BlockSelect<float, int, false, Comparator<float>,
                 NumWarpQ, NumThreadQ, ThreadsPerBlock>
       heap(initK, -1, smemK, smemV, NumWarpQ);

      int row = blockIdx.x;

   // Whole warps must participate in the selection
      int limit = utils::roundDown(num, kWarpSize);
      uint i = threadIdx.x;

      for (; i < limit; i += blockDim.x) {
        int rank= i/k;
        int pos = i%k;
        float* dists1= dists+rank*n*k;
        float dist =dists1[row*k+pos];
        heap.add(dist, i);
      }

      if (i < num) {
        int rank= i/k;
        int pos = i%k;
        float* dists1= dists+rank*n*k;
        float dist =dists1[row*k+pos];
        heap.addThreadQ(dist, i);
      }

      heap.reduce();

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
          outDists[row][i] =smemK[i];
          int v = smemV[i];
           int rank= v/k;
          int pos = v%k;
          faiss::Index::idx_t* nns1= nns+rank*n*k;
          faiss::Index::idx_t label = nns1[row*k+pos];
          outLabels[row][i] = label;
        }
}


void GpuIndexIVFPQ::merge(faiss::Index::idx_t* nns,
                           float* dist,
                           int k,
                           int nq,
                           int nprocess,
                           float* distances,
                           faiss::Index::idx_t* labels) const {
        DeviceScope scope(device_);
      auto stream = resources_->getDefaultStream(device_);
        auto devDistances =
    toDevice<float, 1>(resources_,
                       device_,
                       dist,
                       stream,
                       {(int) k*nq*nprocess});
    auto devLabels =
    toDevice<faiss::Index::idx_t, 1>(resources_,
                                     device_,
                                     nns,
                                     stream,
                                     {(int) k*nq*nprocess});


     auto outDistances =
    toDevice<float, 2>(resources_,
                       device_,
                       distances,
                       stream,
                        {(int) nq, (int) k});
    auto outLabels =
    toDevice<faiss::Index::idx_t, 2>(resources_,
                                     device_,
                                     labels,
                                     stream,
                                    {(int) nq, (int) k});


        constexpr int kThreadsPerBlock = 128;

    auto block = dim3(kThreadsPerBlock);
    auto grid = dim3(nq);
    int num = k*nprocess;
    #define RUN_L2_SELECT(NUM_WARP_Q, NUM_THREAD_Q)                         \
    do {                                                                \
      mergekernel<NUM_WARP_Q, NUM_THREAD_Q, kThreadsPerBlock>       \
        <<<grid, block, 0, stream>>>(devDistances.data(),devLabels.data(), outDistances, \
                                     outLabels,k,nq,nprocess,Limits<float>::getMax());\
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

     fromDevice<float, 2>(
    outDistances, distances, resources_->getDefaultStream(device_));
  fromDevice<faiss::Index::idx_t, 2>(
    outLabels, labels, resources_->getDefaultStream(device_));
}


void
GpuIndexIVFPQ::searchImpl1_(faiss::Index::idx_t n,
                           const float* x,
                           faiss::Index::idx_t k,
                           float* distances,
                           faiss::Index::idx_t* labels) const {
  // Device is already set in GpuIndex::search
   printf("GpuIndexIVFPQ::searchImpl_\n");
  FAISS_ASSERT(index_);
  FAISS_ASSERT(n > 0);

  // Make sure arguments are on the device we desire; use temporary
  // memory allocations to move it if necessary
 auto devX =
    toDevice<float, 2>(resources_,
                       device_,
                       const_cast<float*>(x),
                       resources_->getDefaultStream(device_),
                       {(int) n, (int) index_->getDim()});

   auto devEdgeInfo =
    toDevice<int, 2>(resources_,
                       device_,
                       edgeInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nlist_, (int) numedge_});
    auto devEdgeDistInfo =
    toDevice<float, 2>(resources_,
                       device_,
                       edgeDistInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nlist_, (int) numedge_});
   auto devLambdaInfo =
    toDevice<float, 1>(resources_,
                       device_,
                       lambdaInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nLambda_});

    auto devConstInfo =
    toDevice<float, 1>(resources_,
                       device_,
                       constInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nLambda_});

 index_->queryGraph1(devX,devEdgeInfo,devEdgeDistInfo,devLambdaInfo,devConstInfo,nprobe_,w1_,(int) k,begin_ , end_, (long*)labels);
  //outputVec("devDistances",devDistances.data(),10, resources_->getDefaultStream(device_));
  // Copy back if necessary
}

void
GpuIndexIVFPQ::search1(Index::idx_t n,
                 const float* x,
                 Index::idx_t k,
                 float* distances,
                 Index::idx_t* labels) const {
  DeviceScope scope(device_);

  FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

  if (n > 0) {
      size_t tileSize = 1000;

      for (size_t i = 0; i < n; i += tileSize) {
        size_t curNum = std::min(tileSize, n - i);
        size_t index = i * k ;
        std::cout << "index=   " << index << std::endl;
        searchImpl1_(curNum,
                    x + i * (size_t) this->d,
                    k,
                    distances + index,
                    labels + index);
      }

  }
}

void
GpuIndexIVFPQ::testDistance_(faiss::Index::idx_t n,
                           const float* x,
                           faiss::Index::idx_t k,
                           float* distances,
                           faiss::Index::idx_t* labels) const {
  // Device is already set in GpuIndex::search
  FAISS_ASSERT(index_);
  FAISS_ASSERT(n > 0);

  // Make sure arguments are on the device we desire; use temporary
  // memory allocations to move it if necessary
 auto devX =
    toDevice<float, 2>(resources_,
                       device_,
                       const_cast<float*>(x),
                       resources_->getDefaultStream(device_),
                       {(int) n, (int) index_->getDim()});
 auto devDistances =
    toDevice<float, 2>(resources_,
                       device_,
                       distances,
                       resources_->getDefaultStream(device_),
                       {(int) n, (int) k});
 auto devLabels =
    toDevice<faiss::Index::idx_t, 2>(resources_,
                                     device_,
                                     labels,
                                     resources_->getDefaultStream(device_),
                                     {(int) n, (int) k});
   auto devEdgeInfo =
    toDevice<int, 2>(resources_,
                       device_,
                       edgeInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nlist_, (int) numedge_});
    auto devEdgeDistInfo =
    toDevice<float, 2>(resources_,
                       device_,
                       edgeDistInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nlist_, (int) numedge_});
    auto devLambdaInfo =
    toDevice<float, 1>(resources_,
                       device_,
                       lambdaInfo_,
                       resources_->getDefaultStream(device_),
                       {(int) nLambda_});

 index_->testGraph(devX,devEdgeInfo,devEdgeDistInfo,devLambdaInfo,nprobe_,(int) k,devDistances,devLabels);
  //outputVec("devDistances",devDistances.data(),10, resources_->getDefaultStream(device_));
  // Copy back if necessary
  fromDevice<float, 2>(
    devDistances, distances, resources_->getDefaultStream(device_));
  fromDevice<faiss::Index::idx_t, 2>(
    devLabels, labels, resources_->getDefaultStream(device_));
}


void GpuIndexIVFPQ::writeCodebookToFile(const std::string& _name) {
      std::vector <float> centroids;
      //auto devPQCentroids = index_->getPQCentroids();
       auto devCentroids = quantizer_->getGpuData()->getVectorsFloat32Ref();
       std::vector <float> pq;
      centroids.resize(devCentroids.numElements());
    //pq.resize(devPQCentroids.numElements());
      // fromDevice<float, 3>(devPQCentroids,
       //                  pqData_,
        //                resources_->getDefaultStream(device_));

      fromDevice<float, 2>(devCentroids,
                         centroids.data(),
                         resources_->getDefaultStream(device_));
 int pqCluster = 1 << bitsPerCode_;

    const std::string codebookName = _name + ".ppqt";
    std::ofstream fppqt(codebookName.c_str(), std::ofstream::out | std::ofstream::binary);
	fppqt.write((char*) centroids.data(), devCentroids.numElements()* sizeof(float));
	fppqt.write((char*) pqData_, pqCluster*d* sizeof(float));
	fppqt.write((char*) edgeInfo_,nlist_*numedge_* sizeof(int));
	fppqt.write((char*) edgeDistInfo_, nlist_*numedge_* sizeof(float));
	fppqt.write((char*) lambdaInfo_, nLambda_* sizeof(float));
	fppqt.write((char*) constInfo_, nLambda_* sizeof(float));

    fppqt.close();

}

void GpuIndexIVFPQ::writeCentroidsToFile(const std::string& _name) {
      std::vector <float> centroids;
       auto devCentroids = quantizer_->getGpuData()->getVectorsFloat32Ref();
      centroids.resize(devCentroids.numElements());
      fromDevice<float, 2>(devCentroids,
                         centroids.data(),
                         resources_->getDefaultStream(device_));


    const std::string codebookName = _name + ".umem";
    writeFloat(codebookName,d,nlist_,centroids.data(),0);
}


void GpuIndexIVFPQ::readCodebookFromFile(const std::string& _name) {
    DeviceScope scope(device_);
    const std::string codebookName   =_name+".ppqt";

    std::ifstream f(codebookName.c_str(), std::ofstream::in | std::ofstream::binary);

    float * cb1Host = new float[nlist_ * d];
	f.read((char*) cb1Host, nlist_ * d * sizeof(float));
	quantizer_->reset();
    quantizer_->add(nlist_, cb1Host);
    quantizer_->is_trained = true;
    int pqCluster = 1 << bitsPerCode_;
    pqData_ =new float[pqCluster*d];
    f.read((char*) pqData_,   pqCluster * d * sizeof(float));
    f.read((char*) edgeInfo_,nlist_*numedge_* sizeof(int));
	f.read((char*) edgeDistInfo_, nlist_*numedge_* sizeof(float));
	f.read((char*) lambdaInfo_, nLambda_* sizeof(float));
	f.read((char*) constInfo_, nLambda_* sizeof(float));

     index_ = new IVFPQ(resources_,
                     quantizer_->getGpuData(),
                     numedge_,
                     subQuantizers_,
                     bitsPerCode_,
                     pqData_,
                     false,
                     ivfpqConfig_.indicesOptions,
                     ivfpqConfig_.useFloat16LookupTables,
                     memorySpace_);
  if (reserveMemoryVecs_) {
    index_->reserveMemory(reserveMemoryVecs_);
   }

  index_->setPrecomputedCodes(ivfpqConfig_.usePrecomputedTables);
     this->is_trained = true;
   f.close();
}


void GpuIndexIVFPQ::writeDbToFile(const std::string& _name) {

   const std::string dbName = _name+".dbIdx";
   std::ofstream fdbIdx(dbName.c_str(), std::ofstream::out | std::ofstream::binary);

    const std::string codeName   = _name+".dbcodes";
   std::ofstream fdbcodes(codeName.c_str(), std::ofstream::out | std::ofstream::binary);

    const std::string countName   = _name+".dbcount";
   std::ofstream fdbcount(countName.c_str(), std::ofstream::out | std::ofstream::binary);

    const std::string laName   = _name+".dblas";
   std::ofstream fdblas(laName.c_str(), std::ofstream::out | std::ofstream::binary);
    std::vector <int> counts;
     counts.resize(nlist_*numedge_);

    for (int i = 0; i < nlist_*numedge_; ++i) {

       fdbIdx.write((char*) getListIndices(i).data(), getListIndices(i).size()* sizeof(long));

       fdblas.write((char*) getListLambdas(i).data(), getListLambdas(i).size()* sizeof(uint8_t));

       fdbcodes.write((char*) index_->getListCodesCPU(i).data(),index_->getListCodesCPU(i).size()* sizeof(uint8_t));
       counts[i]  = getListLength(i);
    }
    fdbcount.write((char*)counts.data(), counts.size()* sizeof(int));
    fdbIdx.close();
    fdbcodes.close();
    fdbcount.close();
    fdblas.close();

}


void GpuIndexIVFPQ::readDbFromFile(const std::string& _name) {
   DeviceScope scope(device_);
   const std::string dbName   = _name+".dbIdx";
   std::ifstream fdbIdx(dbName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string codeName   =_name+".dbcodes";
   std::ifstream fdbcodes(codeName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string countName   = _name+".dbcount";
   std::ifstream fdbcount(countName.c_str(), std::ofstream::in | std::ofstream::binary);

   const std::string laName   = _name+".dblas";
   std::ifstream fdblas(laName.c_str(), std::ofstream::in | std::ofstream::binary);

     std::vector < std::vector<long> > ids;
    std::vector < std::vector<uint8_t> > codes;
    std::vector < std::vector<uint8_t> > las;
    std::vector <int> counts;
     ids.resize(nlist_*numedge_);
     las.resize(nlist_*numedge_);
     codes.resize(nlist_*numedge_);
     counts.resize(nlist_*numedge_);
     fdbcount.read((char*)counts.data(), counts.size()* sizeof(int));
    for (int i = 0; i < nlist_*numedge_; ++i) {
       int length = counts[i];
       index_->deviceListLengths_[i] = length;
       }
     int tmpl =0;
    for (int i = 0; i < nlist_*numedge_; ++i) {
       int length = counts[i];
       tmpl+=length;
       ids[i].resize(length);
       las[i].resize(length);
       codes[i].resize(length*subQuantizers_);
       fdbIdx.read((char*) ids[i].data(), ids[i].size()* sizeof(long));
       fdblas.read((char*) las[i].data(), las[i].size()* sizeof(uint8_t));
       fdbcodes.read((char*) codes[i].data(),codes[i].size()* sizeof(uint8_t));
       if(i%10000==0){
          std::cout <<"i: "<< i<<" length: "<< tmpl<< std::endl;
       }
       index_->addCodeVectorsFromCpu(i, codes[i].data(), ids[i].data(), ids[i].size());
       index_->addLambdasFromCpu(i, las[i].data(), las[i].size());
    }

     std::cout <<"ListLength"<< std::endl;
  	for (int i = 0; i <getListLength(135*numedge_+1); i++){
  	  std::cout << "\t" << getListIndices(135*numedge_+1)[i];
  	  std::cout << "\t" <<(int)getListLambdas(135*numedge_+1)[i];
  	}

	std::cout << std::endl;
	std::cout << std::endl;

    fdbIdx.close();
    fdbcodes.close();
    fdbcount.close();
    fdblas.close();
}
__global__ void outputVecIntKernel( void** listCodes, void** listLambdas,
                           int* listLengths,
                          int i) {
	if (threadIdx.x == 0) {
      int length= listLengths[i];
      unsigned char* codeList = (unsigned char*) listCodes[i];

      uint8_t* lambdaList = (uint8_t*) listLambdas[i];
      printf("length:%d",length);
      printf("codeList:%d,lambdaList:%d",codeList[0],lambdaList[0]);

	}
}
void GpuIndexIVFPQ::readDbFromFile(const std::string& _name,size_t nb) {
    DeviceScope scope(device_);

    auto stream = resources_->getDefaultStream(device_);

   const std::string dbName   = _name+".dbIdx";
   std::ifstream fdbIdx(dbName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string codeName   =_name+".dbcodes";
   std::ifstream fdbcodes(codeName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string countName   = _name+".dbcount";
   std::ifstream fdbcount(countName.c_str(), std::ofstream::in | std::ofstream::binary);

   const std::string laName   = _name+".dblas";
   std::ifstream fdblas(laName.c_str(), std::ofstream::in | std::ofstream::binary);

     std::vector < std::vector<long> > ids;
     int add_bs = 2000000;
    uint8_t* codes;
    uint8_t* las;
    std::vector <int> counts;
    counts.resize(nlist_*numedge_);
     ids.resize(nlist_*numedge_);
    fdbcount.read((char*)counts.data(), counts.size()* sizeof(int));


     codes = new uint8_t[add_bs*subQuantizers_];
     las = new uint8_t[add_bs];
     uint8_t* d_codes;
     uint8_t* d_las;
     size_t tsize = nb*subQuantizers_;
     allocMemorySpace(memorySpace_, (void**)&d_codes, tsize* sizeof(uint8_t));
     allocMemorySpace(memorySpace_, (void**)&d_las, nb* sizeof(uint8_t));
     for (size_t begin = 0; begin <nb; begin += add_bs) {
        size_t chunksize = min ((long)add_bs, nb-begin);
        fdblas.read((char*)las, chunksize* sizeof(uint8_t));
        fdbcodes.read((char*)codes,chunksize*subQuantizers_*sizeof(uint8_t));
        CUDA_VERIFY(cudaMemcpyAsync(d_codes + begin*subQuantizers_, codes, chunksize*subQuantizers_ * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_VERIFY(cudaMemcpyAsync(d_las + begin, las, chunksize * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice, stream));

    }
     size_t total =0;
     for (int i = 0; i < nlist_*numedge_; ++i) {
        if(i>0)
        total+=counts[i-1];

        index_->deviceListLengths_[i] = counts[i];
        index_->maxListLength_ = max(index_->maxListLength_, counts[i]);

        auto& listCodes = index_->deviceListData_[i];
        auto& listLas = index_->deviceListLambdas_[i];
        size_t offset = total*subQuantizers_;
        listCodes->data_=d_codes+offset;
        listCodes->num_=counts[i]*subQuantizers_;
        listCodes->capacity_=counts[i]*subQuantizers_;
        index_->deviceListDataPointers_[i] = listCodes->data();
        listLas->data_=d_las+total;
        listLas->num_=counts[i];
        listLas->capacity_=counts[i];
        index_->deviceListLambdaPointers_[i] = listLas->data();
        int length = counts[i];
       ids[i].resize(length);

       fdbIdx.read((char*) ids[i].data(), ids[i].size()* sizeof(long));

       index_->addIndicesFromCpu_(i, ids[i].data(), ids[i].size());

        if(i%100000==0){
          std::cout <<"i: "<< i<<" total: "<< total<< std::endl;

         //   dim3 grid(1, 1, 1);
         //   dim3 block(16, 1, 1);
         //   outputVecIntKernel<<<grid, block,0,stream>>>(index_->deviceListDataPointers_.data().get(),
          //                           index_->deviceListLambdaPointers_.data().get(),
         //                             index_->deviceListLengths_.data().get(),
           //                         i);

       }
      }
   std::cout <<"total: "<<total << std::endl;
      // std::cout <<"ListLength"<<getListLength(570000) << std::endl;

	std::cout << std::endl;
	std::cout << std::endl;

    fdbIdx.close();
    fdbcodes.close();
    fdbcount.close();
    fdblas.close();
}


void GpuIndexIVFPQ::readDbFromFile1(const std::string& _name) {

   const std::string dbName   = _name+".dbIdx";
   std::ifstream fdbIdx(dbName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string codeName   =_name+".dbcodes";
   std::ifstream fdbcodes(codeName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string countName   = _name+".dbcount";
   std::ifstream fdbcount(countName.c_str(), std::ofstream::in | std::ofstream::binary);

   const std::string laName   = _name+".dblas";
   std::ifstream fdblas(laName.c_str(), std::ofstream::in | std::ofstream::binary);

     std::vector < std::vector<long> > ids;
    std::vector <int> counts;
     ids.resize(nlist_*numedge_);
     counts.resize(nlist_*numedge_);
     fdbcount.read((char*)counts.data(), counts.size()* sizeof(int));
     std::vector<uint> binHist;
	binHist.resize(8);
	uint total = 0;

	uint maxVal = 0;
	uint maxIdx = 0;
    for (int i = 0; i < nlist_*numedge_; ++i) {
       int length = counts[i];
       ids[i].resize(length);

       fdbIdx.read((char*) ids[i].data(), ids[i].size()* sizeof(long));

       index_->addIndicesFromCpu_(i, ids[i].data(), ids[i].size());

       if (counts[i] > maxVal) {
			maxVal = counts[i];
			maxIdx = i;
		}

        if (counts[i] == 0)
			binHist[0]++;
		else if (counts[i] < 5)
			binHist[1]++;
		else if (counts[i] < 10)
			binHist[2]++;
		else if (counts[i] < 20)
			binHist[3]++;
		else if (counts[i] < 50)
			binHist[4]++;
		else if (counts[i] < 100)
			binHist[5]++;
		else if (counts[i] < 500)
			binHist[6]++;
		else
			binHist[7]++;

		total += counts[i];



    }

    cout << "total entries: " << total << endl;

	 std::cout << "histogram: " << endl;
	 std::cout << "0 \t" << binHist[0] << endl;
	 std::cout << "<5 \t" << binHist[1] << endl;
	 std::cout << "<10 \t" << binHist[2] << endl;
	 std::cout << "<20 \t" << binHist[3] << endl;
	 std::cout << "<50\t" << binHist[4] << endl;
	 std::cout << "<100 \t" << binHist[5] << endl;
	 std::cout << "<500 \t" << binHist[6] << endl;
	 std::cout << ">500 \t" << binHist[7] << endl;

	 std::cout << "maxbin: " << maxIdx << "  entries: " << maxVal << endl;

     for(int a =0 ; a<8;a++){
                 std::cout << binHist[a] << ",";
        }
    std::cout << std::endl;
     std::cout <<"ListLength"<< std::endl;
  	for (int i = 0; i <getListLength(135*numedge_+1); i++){
  	  std::cout << "\t" << getListIndices(135*numedge_+1)[i];
  	}

	std::cout << std::endl;

    fdbIdx.close();
    fdbcodes.close();
    fdbcount.close();
    fdblas.close();
}


void GpuIndexIVFPQ::readDbFromFile(const std::string& _name,size_t nb,int pronum,int rank) {
    DeviceScope scope(device_);

    auto stream = resources_->getDefaultStream(device_);

   const std::string dbName   = _name+".dbIdx";
   std::ifstream fdbIdx(dbName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string codeName   =_name+".dbcodes";
   std::ifstream fdbcodes(codeName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string countName   = _name+".dbcount";
   std::ifstream fdbcount(countName.c_str(), std::ofstream::in | std::ofstream::binary);

   const std::string laName   = _name+".dblas";
   std::ifstream fdblas(laName.c_str(), std::ofstream::in | std::ofstream::binary);

     std::vector < std::vector<long> > ids;
     int add_bs = 2000000;
    uint8_t* codes;
    uint8_t* las;
    std::vector <int> counts;
    counts.resize(nlist_*numedge_);
     ids.resize(nlist_*numedge_);
    fdbcount.read((char*)counts.data(), counts.size()* sizeof(int));

    begin_=(nlist_/pronum)*rank;
    end_=begin_+nlist_/pronum-1;

    if(rank==pronum-1)
    {
        end_=nlist_-1;
    }

    int start=(nlist_*numedge_/pronum)*rank;
    int end=start+nlist_*numedge_/pronum-1;

    size_t startoffset=0;
    size_t endoffset=0;
    for(int i=0; i < start; ++i)
    {
      startoffset+=counts[i];
      endoffset+=counts[i];
    }
    for(int i=start;i<end+1;i++)
    {
      endoffset+=counts[i];
    }

    nb=endoffset-startoffset;

    for(int i = 0; i < nlist_*numedge_; ++i)
    {
        if(i<start||i>end)
       {
        counts[i]=0;
       }
    }

    std::cout<<"--------------startoffset------------:"<<startoffset<<std::endl;
    std::cout<<"--------------endoffset------------:"<<endoffset<<std::endl;
    std::cout<<"--------------start------------:"<<start<<std::endl;
    std::cout<<"--------------end------------:"<<end<<std::endl;

     codes = new uint8_t[add_bs*subQuantizers_];
     las = new uint8_t[add_bs];
     uint8_t* d_codes;
     uint8_t* d_las;
     size_t tsize = nb*subQuantizers_;
     allocMemorySpace(memorySpace_, (void**)&d_codes, tsize* sizeof(uint8_t));
     allocMemorySpace(memorySpace_, (void**)&d_las, nb* sizeof(uint8_t));

     fdblas.seekg(startoffset*sizeof(uint8_t), std::ios::beg);
     fdbcodes.seekg(startoffset*subQuantizers_*sizeof(uint8_t),std::ios::beg);
     for (size_t begin = 0; begin <nb; begin += add_bs) {
        size_t chunksize = min ((long)add_bs, nb-begin);
        fdblas.read((char*)las, chunksize* sizeof(uint8_t));
        fdbcodes.read((char*)codes,chunksize*subQuantizers_*sizeof(uint8_t));
        CUDA_VERIFY(cudaMemcpyAsync(d_codes + begin*subQuantizers_, codes, chunksize*subQuantizers_ * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_VERIFY(cudaMemcpyAsync(d_las + begin, las, chunksize * sizeof(uint8_t),
                                    cudaMemcpyHostToDevice, stream));

    }
     size_t total =0;
     fdbIdx.seekg(startoffset*sizeof(long),std::ios::beg);
     for (int i = 0; i < nlist_*numedge_; ++i) {
        if(i>0)
        total+=counts[i-1];

        index_->deviceListLengths_[i] = counts[i];
        index_->maxListLength_ = max(index_->maxListLength_, counts[i]);

        auto& listCodes = index_->deviceListData_[i];
        auto& listLas = index_->deviceListLambdas_[i];


        size_t offset = total*subQuantizers_;
        listCodes->data_=d_codes+offset;
        listCodes->num_=counts[i]*subQuantizers_;
        listCodes->capacity_=counts[i]*subQuantizers_;

        listLas->data_=d_las+total;
        listLas->num_=counts[i];
        listLas->capacity_=counts[i];



        int length = counts[i];
       ids[i].resize(length);

       //fdbIdx.seek();
       fdbIdx.read((char*) ids[i].data(), ids[i].size()* sizeof(long));



       index_->deviceListDataPointers_[i] = listCodes->data();
       index_->deviceListLambdaPointers_[i] = listLas->data();

       index_->addIndicesFromCpu_(i, ids[i].data(), ids[i].size());

        if(i%10000==0){
          std::cout <<"i: "<< i<<" total: "<< total<< std::endl;

       }
      }

       std::cout <<"ListLength"<<getListLength(570000) << std::endl;

  std::cout << std::endl;
  std::cout << std::endl;

    fdbIdx.close();
    fdbcodes.close();
    fdbcount.close();
    fdblas.close();
}






void GpuIndexIVFPQ::readDbFromFile(const std::string& _name,int pronum,int rank) {

   const std::string dbName   = _name+".dbIdx";
   std::ifstream fdbIdx(dbName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string codeName   =_name+".dbcodes";
   std::ifstream fdbcodes(codeName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string countName   = _name+".dbcount";
   std::ifstream fdbcount(countName.c_str(), std::ofstream::in | std::ofstream::binary);

   const std::string laName   = _name+".dblas";
   std::ifstream fdblas(laName.c_str(), std::ofstream::in | std::ofstream::binary);

     std::vector < std::vector<long> > ids;
    std::vector < std::vector<uint8_t> > codes;
    std::vector < std::vector<uint8_t> > las;
    std::vector <int> counts;
     ids.resize(nlist_*numedge_);
     las.resize(nlist_*numedge_);
     codes.resize(nlist_*numedge_);
     counts.resize(nlist_*numedge_);
     fdbcount.read((char*)counts.data(), counts.size()* sizeof(int));

     int start=(nlist_*numedge_/pronum)*rank;
     int end=start+nlist_*numedge_/pronum-1;

    for (int i = 0; i < nlist_*numedge_; ++i) {
       int length = counts[i];
       ids[i].resize(length);
       las[i].resize(length);
       codes[i].resize(length*subQuantizers_);
       fdbIdx.read((char*) ids[i].data(), ids[i].size()* sizeof(long));
       fdblas.read((char*) las[i].data(), las[i].size()* sizeof(uint8_t));
       fdbcodes.read((char*) codes[i].data(),codes[i].size()* sizeof(uint8_t));

       if(i<start||i>end)
       {
           ids[i].resize(0);
           codes[i].resize(0);
           las[i].resize(0);
       }

       index_->addCodeVectorsFromCpu(i, codes[i].data(), ids[i].data(), ids[i].size());
       index_->addLambdasFromCpu(i, las[i].data(), las[i].size());
    }

     std::cout <<"ListLength"<< std::endl;
  	for (int i = 0; i <getListLength(135*numedge_+1); i++){
  	  std::cout << "\t" << getListIndices(135*numedge_+1)[i];
  	  std::cout << "\t" <<(int)getListLambdas(135*numedge_+1)[i];
  	}

	std::cout << std::endl;
	std::cout << std::endl;

    fdbIdx.close();
    fdbcodes.close();
    fdbcount.close();
    fdblas.close();
}








int
GpuIndexIVFPQ::getListLength(int listId) const {
  FAISS_ASSERT(index_);
  return index_->getListLength(listId);
}

std::vector<unsigned char>
GpuIndexIVFPQ::getListCodes(int listId) const {
  FAISS_ASSERT(index_);
  DeviceScope scope(device_);

  return index_->getListCodes(listId);
}

std::vector<unsigned char>
GpuIndexIVFPQ::getListLambdas(int listId)const  {
  FAISS_ASSERT(index_);
  DeviceScope scope(device_);

  return index_->getListLambdas(listId);
}

std::vector<long>
GpuIndexIVFPQ::getListIndices(int listId) const {
  FAISS_ASSERT(index_);
  DeviceScope scope(device_);

  return index_->getListIndices(listId);
}

void
GpuIndexIVFPQ::verifySettings_() const {
  // Our implementation has these restrictions:

  // Must have some number of lists
  FAISS_THROW_IF_NOT_MSG(nlist_ > 0, "nlist must be >0");

  // up to a single byte per code
  FAISS_THROW_IF_NOT_FMT(bitsPerCode_ <= 8,
                     "Bits per code must be <= 8 (passed %d)", bitsPerCode_);

  // Sub-quantizers must evenly divide dimensions available
  FAISS_THROW_IF_NOT_FMT(this->d % subQuantizers_ == 0,
                     "Number of sub-quantizers (%d) must be an "
                     "even divisor of the number of dimensions (%d)",
                     subQuantizers_, this->d);

  // The number of bytes per encoded vector must be one we support
  FAISS_THROW_IF_NOT_FMT(IVFPQ::isSupportedPQCodeLength(subQuantizers_),
                     "Number of bytes per encoded vector / sub-quantizers (%d) "
                     "is not supported",
                     subQuantizers_);

  // We must have enough shared memory on the current device to store
  // our lookup distances
  int lookupTableSize = sizeof(float);
#ifdef FAISS_USE_FLOAT16
  if (ivfpqConfig_.useFloat16LookupTables) {
    lookupTableSize = sizeof(half);
  }
#endif

  // 64 bytes per code is only supported with usage of float16, at 2^8
  // codes per subquantizer
  size_t requiredSmemSize =
    lookupTableSize * subQuantizers_ * utils::pow2(bitsPerCode_);
  size_t smemPerBlock = getMaxSharedMemPerBlock(device_);

////  FAISS_THROW_IF_NOT_FMT(requiredSmemSize
   //                  <= getMaxSharedMemPerBlock(device_),
  //                   "Device %d has %zu bytes of shared memory, while "
     //                "%d bits per code and %d sub-quantizers requires %zu "
    //                 "bytes. Consider useFloat16LookupTables and/or "
       //              "reduce parameters",
       //              device_, smemPerBlock, bitsPerCode_, subQuantizers_,
       //              requiredSmemSize);

  // If precomputed codes are disabled, we have an extra limitation in
  // terms of the number of dimensions per subquantizer
  FAISS_THROW_IF_NOT_FMT(ivfpqConfig_.usePrecomputedTables ||
                     IVFPQ::isSupportedNoPrecomputedSubDimSize(
                       this->d / subQuantizers_),
                     "Number of dimensions per sub-quantizer (%d) "
                     "is unsupported with precomputed codes",
                     this->d / subQuantizers_);

  // TODO: fully implement METRIC_INNER_PRODUCT
  FAISS_THROW_IF_NOT_MSG(this->metric_type == faiss::METRIC_L2,
                     "METRIC_INNER_PRODUCT is currently unsupported");
}

} } // namespace
