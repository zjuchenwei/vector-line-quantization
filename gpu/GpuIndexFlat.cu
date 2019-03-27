/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
#include <iostream>
#include "GpuIndexFlat.h"
#include "../IndexFlat.h"
#include "GpuResources.h"
#include "impl/FlatIndex.cuh"
#include "utils/CopyUtils.cuh"
#include "utils/helper.cuh"
#include "utils/DeviceUtils.h"
#include "utils/Float16.cuh"
#include "utils/triangle.cuh"
#include "utils/StaticUtils.h"
#include "utils/DeviceDefs.cuh"
#include "utils/PtxUtils.cuh"
#include "utils/MathOperators.cuh"
#include "utils/Reductions.cuh"
#include "utils/bitonicSort.cuh"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <limits>

namespace faiss { namespace gpu {

/// Default CPU search size for which we use paged copies
constexpr size_t kMinPageSize = (size_t) 256 * 1024 * 1024;

/// Size above which we page copies from the CPU to GPU (non-paged
/// memory usage)
constexpr size_t kNonPinnedPageSize = (size_t) 256 * 1024 * 1024;

GpuIndexFlat::GpuIndexFlat(GpuResources* resources,
                           const faiss::IndexFlat* index,
                           GpuIndexFlatConfig config) :
    GpuIndex(resources, index->d, index->metric_type, config),
    minPagedSize_(kMinPageSize),
    config_(config),
    data_(nullptr) {
  verifySettings_();

  // Flat index doesn't need training
  this->is_trained = true;

  copyFrom(index);
}

GpuIndexFlat::GpuIndexFlat(GpuResources* resources,
                           int dims,
                           faiss::MetricType metric,
                           GpuIndexFlatConfig config) :
    GpuIndex(resources, dims, metric, config),
    minPagedSize_(kMinPageSize),
    config_(config),
    data_(nullptr) {
  verifySettings_();

  // Flat index doesn't need training
  this->is_trained = true;

  // Construct index
  DeviceScope scope(device_);
  data_ = new FlatIndex(resources,
                        dims,
                        metric == faiss::METRIC_L2,
                        config_.useFloat16,
                        config_.useFloat16Accumulator,
                        config_.storeTransposed,
                        memorySpace_);
}

GpuIndexFlat::~GpuIndexFlat() {
  delete data_;
}

void
GpuIndexFlat::setMinPagingSize(size_t size) {
  minPagedSize_ = size;
}

size_t
GpuIndexFlat::getMinPagingSize() const {
  return minPagedSize_;
}

void
GpuIndexFlat::copyFrom(const faiss::IndexFlat* index) {
  DeviceScope scope(device_);

  this->d = index->d;
  this->metric_type = index->metric_type;

  // GPU code has 32 bit indices
  FAISS_THROW_IF_NOT_FMT(index->ntotal <=
                     (faiss::Index::idx_t) std::numeric_limits<int>::max(),
                     "GPU index only supports up to %zu indices; "
                     "attempting to copy CPU index with %zu parameters",
                     (size_t) std::numeric_limits<int>::max(),
                     (size_t) index->ntotal);
  this->ntotal = index->ntotal;

  delete data_;
  data_ = new FlatIndex(resources_,
                        this->d,
                        index->metric_type == faiss::METRIC_L2,
                        config_.useFloat16,
                        config_.useFloat16Accumulator,
                        config_.storeTransposed,
                        memorySpace_);

  // The index could be empty
  if (index->ntotal > 0) {
    data_->add(index->xb.data(),
               index->ntotal,
               resources_->getDefaultStream(device_));
  }
}

void
GpuIndexFlat::copyTo(faiss::IndexFlat* index) const {
  DeviceScope scope(device_);

  index->d = this->d;
  index->ntotal = this->ntotal;
  index->metric_type = this->metric_type;

  FAISS_ASSERT(data_->getSize() == this->ntotal);
  index->xb.resize(this->ntotal * this->d);

  auto stream = resources_->getDefaultStream(device_);

  if (this->ntotal > 0) {
    if (config_.useFloat16) {
      auto vecFloat32 = data_->getVectorsFloat32Copy(stream);
      fromDevice(vecFloat32, index->xb.data(), stream);
    } else {
      fromDevice(data_->getVectorsFloat32Ref(), index->xb.data(), stream);
    }
  }
}

size_t
GpuIndexFlat::getNumVecs() const {
  return this->ntotal;
}

void
GpuIndexFlat::reset() {
  DeviceScope scope(device_);

  // Free the underlying memory
  data_->reset();
  this->ntotal = 0;
}

void
GpuIndexFlat::train(Index::idx_t n, const float* x) {
  // nothing to do
   printf ("GpuIndexFlat::train\n");
}

void
GpuIndexFlat::add(Index::idx_t n, const float* x) {
  DeviceScope scope(device_);
 printf ("GpuIndexFlat::add\n");
  // To avoid multiple re-allocations, ensure we have enough storage
  // available
  data_->reserve(n, resources_->getDefaultStream(device_));

  // If we're not operating in float16 mode, we don't need the input
  // data to be resident on our device; we can add directly.
  if (!config_.useFloat16) {
   printf ("GpuIndexFlat::add111\n");
    addImpl_(n, x, nullptr);
  } else {
    // Otherwise, perform the paging
    GpuIndex::add(n, x);
  }
}

void



GpuIndexFlat::addImpl_(Index::idx_t n,
                       const float* x,
                       const Index::idx_t* ids) {
  // Device is already set in GpuIndex::addInternal_

  // We do not support add_with_ids
  FAISS_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported");
  FAISS_THROW_IF_NOT(n > 0);

  // Due to GPU indexing in int32, we can't store more than this
  // number of vectors on a GPU
  FAISS_THROW_IF_NOT_FMT(this->ntotal + n <=
                     (faiss::Index::idx_t) std::numeric_limits<int>::max(),
                     "GPU index only supports up to %zu indices",
                     (size_t) std::numeric_limits<int>::max());

  data_->add(x, n, resources_->getDefaultStream(device_));
  this->ntotal += n;
}

struct IntToLong {
  __device__ long operator()(int v) const { return (long) v; }
};

void
GpuIndexFlat::search(faiss::Index::idx_t n,
                     const float* x,
                     faiss::Index::idx_t k,
                     float* distances,
                     faiss::Index::idx_t* labels) const {
  if (n == 0) {
    return;
  }
  printf("GpuIndexFlat::search\n");
  // For now, only support <= max int results
  FAISS_THROW_IF_NOT_FMT(n <=
                     (faiss::Index::idx_t) std::numeric_limits<int>::max(),
                     "GPU index only supports up to %zu indices",
                     (size_t) std::numeric_limits<int>::max());
  FAISS_THROW_IF_NOT_FMT(k <= 1024,
                     "GPU only supports k <= 1024 (requested %d)",
                     (int) k); // select limitation

  DeviceScope scope(device_);
  auto stream = resources_->getDefaultStream(device_);

  // The input vectors may be too large for the GPU, but we still
  // assume that the output distances and labels are not.
  // Go ahead and make space for output distances and labels on the
  // GPU.
  // If we reach a point where all inputs are too big, we can add
  // another level of tiling.

  auto outDistances = toDevice<float, 2>(resources_,
                                         device_,
                                         distances,
                                         stream,
                                         {(int) n, (int) k});

  // FlatIndex only supports an interface returning int indices
  DeviceTensor<int, 2, true> outIntIndices(
    resources_->getMemoryManagerCurrentDevice(),
    {(int) n, (int) k}, stream);

  bool usePaged = false;

  if (getDeviceForAddress(x) == -1) {
    // It is possible that the user is querying for a vector set size
    // `x` that won't fit on the GPU.
    // In this case, we will have to handle paging of the data from CPU
    // -> GPU.
    // Currently, we don't handle the case where the output data won't
    // fit on the GPU (e.g., n * k is too large for the GPU memory).
    size_t dataSize = (size_t) n * this->d * sizeof(float);

    if (dataSize >= minPagedSize_) {
      searchFromCpuPaged_(n, x, k,
                          outDistances.data(),
                          outIntIndices.data());
      usePaged = true;
    }
  }

  if (!usePaged) {
    searchNonPaged_(n, x, k,
                    outDistances.data(),
                    outIntIndices.data());
  }

  // Convert and copy int indices out
  auto outIndices = toDevice<faiss::Index::idx_t, 2>(resources_,
                                                     device_,
                                                     labels,
                                                     stream,
                                                     {(int) n, (int) k});

  // Convert int to long
  thrust::transform(thrust::cuda::par.on(stream),
                    outIntIndices.data(),
                    outIntIndices.end(),
                    outIndices.data(),
                    IntToLong());

  // Copy back if necessary
  fromDevice<float, 2>(outDistances, distances, stream);
  fromDevice<faiss::Index::idx_t, 2>(outIndices, labels, stream);
}

void
GpuIndexFlat::searchInt(faiss::Index::idx_t n,
                     const float* x,
                     faiss::Index::idx_t k,
                     float* distances,
                     int* labels) const {
  if (n == 0) {
    return;
  }
  printf("GpuIndexFlat::search\n");
  // For now, only support <= max int results
  FAISS_THROW_IF_NOT_FMT(n <=
                     (faiss::Index::idx_t) std::numeric_limits<int>::max(),
                     "GPU index only supports up to %zu indices",
                     (size_t) std::numeric_limits<int>::max());
  FAISS_THROW_IF_NOT_FMT(k <= 1024,
                     "GPU only supports k <= 1024 (requested %d)",
                     (int) k); // select limitation

  DeviceScope scope(device_);
  auto stream = resources_->getDefaultStream(device_);

  // The input vectors may be too large for the GPU, but we still
  // assume that the output distances and labels are not.
  // Go ahead and make space for output distances and labels on the
  // GPU.
  // If we reach a point where all inputs are too big, we can add
  // another level of tiling.

  auto outDistances = toDevice<float, 2>(resources_,
                                         device_,
                                         distances,
                                         stream,
                                         {(int) n, (int) k});

    // Convert and copy int indices out
  auto outIndices = toDevice<int, 2>(resources_,
                                                     device_,
                                                     labels,
                                                     stream,
                                                     {(int) n, (int) k});

  bool usePaged = false;

  if (getDeviceForAddress(x) == -1) {
    // It is possible that the user is querying for a vector set size
    // `x` that won't fit on the GPU.
    // In this case, we will have to handle paging of the data from CPU
    // -> GPU.
    // Currently, we don't handle the case where the output data won't
    // fit on the GPU (e.g., n * k is too large for the GPU memory).
    size_t dataSize = (size_t) n * this->d * sizeof(float);

    if (dataSize >= minPagedSize_) {
      searchFromCpuPaged_(n, x, k,
                          outDistances.data(),
                          outIndices.data());
      usePaged = true;
    }
  }

  if (!usePaged) {
    searchNonPaged_(n, x, k,
                    outDistances.data(),
                    outIndices.data());
  }




  // Copy back if necessary
  fromDevice<float, 2>(outDistances, distances, stream);
  fromDevice<int, 2>(outIndices, labels, stream);
}


void
GpuIndexFlat::buildGraph(faiss::Index::idx_t n,
                     int k,
                     float* distances,
                     int* labels) const {
  if (n == 0) {
    return;
  }
  printf("GpuIndexFlat::buildGraph\n");
  // For now, only support <= max int results
  FAISS_THROW_IF_NOT_FMT(n <=
                     (faiss::Index::idx_t) std::numeric_limits<int>::max(),
                     "GPU index only supports up to %zu indices",
                     (size_t) std::numeric_limits<int>::max());
  FAISS_THROW_IF_NOT_FMT(k <= 1024,
                     "GPU only supports k <= 1024 (requested %d)",
                     (int) k); // select limitation

  DeviceScope scope(device_);
  auto stream = resources_->getDefaultStream(device_);

  // The input vectors may be too large for the GPU, but we still
  // assume that the output distances and labels are not.
  // Go ahead and make space for output distances and labels on the
  // GPU.
  // If we reach a point where all inputs are too big, we can add
  // another level of tiling.
  auto outDistances = toDevice<float, 2>(resources_,
                                         device_,
                                         distances,
                                         stream,
                                         {(int) n, (int) k});


  bool usePaged = false;


  // Convert and copy int indices out
  auto outIndices = toDevice<int, 2>(resources_,
                                                     device_,
                                                     labels,
                                                     stream,
                                                     {(int) n, (int) k});

 if (!usePaged) {
    buildGraphNonPaged_(n, k,
                    outDistances.data(),
                    outIndices.data());
  }


  // Copy back if necessary
  fromDevice<float, 2>(outDistances, distances, stream);
  fromDevice<int, 2>(outIndices, labels, stream);
}



template <typename T, typename TVec,
          int RowTileSize>
__global__ void get1BinKernel_nms( Tensor<int, 2, true> assign2,Tensor<float, 2, true> d_Mf, Tensor<int, 2, true> assign1,
       Tensor<TVec, 2, true> _cb,
       Tensor<TVec, 2, true> input,Tensor<float, 2, true> edgedistinfo,
       Tensor<int, 2, true> edgeinfo) {
    //printf("sdfsadfsad\n");
    extern __shared__ char smemByte[];
    T* smem = (T*) smemByte;
    int numWarps = utils::divUp(blockDim.x, kWarpSize);
    int d_edge = edgeinfo.getSize(1);
    T* shmIter = (T*) smemByte;
    shmIter+= numWarps;
    float* dist = shmIter;
    shmIter +=  (d_edge+1);
    float* dist1 = shmIter;
    shmIter += d_edge;
    float* lamda = shmIter;
    shmIter += d_edge;
    uint* ids = (uint*) shmIter;


    int laneId = getLaneId();
    int warpId = threadIdx.x / kWarpSize;
    int RowTileSize1= RowTileSize;
    bool lastRowTile = (blockIdx.x == (gridDim.x - 1));
    int rowStart = RowTileSize * blockIdx.x;
    float tmp;

    if (lastRowTile) {
        RowTileSize1 = input.getSize(0) - rowStart;
    }
    // We are handling the very end of the input matrix rows
    for (int row = 0; row < RowTileSize1; ++row) {

        TVec val = input[rowStart + row][threadIdx.x];
         int A=assign1[rowStart + row][0];



        for (int a = 0; a < d_edge; a++) {
            int s = edgeinfo[A][a];
            TVec val1 = _cb[s][threadIdx.x];
            TVec tmpval = Math<TVec>::sub(val, val1);
            TVec tmpval1= Math<TVec>::mul(tmpval,tmpval);
            tmp=Math<TVec>::reduceAdd(tmpval1);
            tmp=warpReduceAllSum(tmp);
            if (laneId == 0) {
            smem[warpId] = tmp;
            }

            // Sum across warps
            if (warpId == 0) {
                tmp = laneId < numWarps ?
                              smem[laneId] : Math<T>::zero();
                tmp = warpReduceAllSum(tmp);
                if (laneId == 0) {
                    dist[a] = tmp;
                }
            }

       }
        __syncthreads();

        TVec val2 = _cb[A][threadIdx.x];
        TVec tmpval = Math<TVec>::sub(val, val2);
        TVec tmpval1= Math<TVec>::mul(tmpval,tmpval);
        tmp=Math<TVec>::reduceAdd(tmpval1);
        tmp=warpReduceAllSum(tmp);
        if (laneId == 0) {
            smem[warpId] = tmp;
        }

        // Sum across warps
        if (warpId == 0) {
            tmp = laneId < numWarps ?
                          smem[laneId] : Math<T>::zero();
            tmp = warpReduceAllSum(tmp);
             if (laneId == 0) {
                dist[d_edge] = tmp;
            }

        }
        __syncthreads();
        for (int iter1 = threadIdx.x; iter1 < d_edge; iter1 += blockDim.x) {
            float a=dist[iter1];
            float b=dist[d_edge];
            float c=edgedistinfo[A][iter1];


            float lambda11=project(a,b,c);
            float q2=dist2(a,b,c,lambda11);
            __syncthreads();

            dist1[iter1] = q2;
            lamda[iter1] = lambda11;
            ids[iter1]=iter1;
        }
          __syncthreads();
        bitonic3(dist1, ids, d_edge);

        if(threadIdx.x==0){
            int i=0;
              uint edgeid = ids[0];
              float la = lamda[ids[0]];
        	for(;i<d_edge;i++)
			{
				if(lamda[ids[i]]>=0&&lamda[ids[i]]<=1)
				{
                    edgeid = ids[i];
                    la = lamda[ids[i]];
					break;
				}
			}

            assign2[rowStart + row][0]=A*d_edge+edgeid;
            d_Mf[rowStart + row][0]=la;
        }
        __syncthreads();
        //printf("asdfsd\n");
    }



}

template <int RowTileSize>
__global__ void assignLambdaKernel( Tensor<float, 1, true> lambdaf,Tensor<uint8_t, 1, true> lambda,
       Tensor<float, 1, true> lambdainfo) {
    extern __shared__ char smemByte[];
    float* smem = (float*) smemByte;
    int d_edge = lambdainfo.getSize(0);
    float* shmIter = (float*) smemByte;
    float* dist = shmIter;
    shmIter +=  (d_edge);
    uint* ids = (uint*) shmIter;

    int RowTileSize1= RowTileSize;
    bool lastRowTile = (blockIdx.x == (gridDim.x - 1));
    int rowStart = RowTileSize * blockIdx.x;
    float tmp;

    if (lastRowTile) {
        RowTileSize1 = lambdaf.getSize(0) - rowStart;
    }
    // We are handling the very end of the input matrix rows
    for (int row = 0; row < RowTileSize1; ++row) {

        float val = lambdaf[rowStart + row];

      for(int iter = threadIdx.x; iter<lambdainfo.getSize(0);iter+=blockDim.x){
          float tmp = val-lambdainfo[iter];
          dist[iter] = Math<float>::mul(tmp,tmp);
          ids[iter]=iter;
      }

        __syncthreads();

       bitonic3(dist, ids, d_edge);

        if(threadIdx.x==0){
           lambda[rowStart + row]=ids[0];
        }
        __syncthreads();
        //printf("asdfsd\n");
    }



}



void
GpuIndexFlat::assign1(
      faiss::Index::idx_t n,
      int dim,
      const float* x,
      int* assign1,
      int* assign2,
      float* lamdaf,
      int* edgeinfo,
      float* edgedistinfo,int nlist,int numedge,int k)const {
  if (n == 0) {
    return;
  }
  printf("nlist : %d ,numedge : %d ,n  :%d,k:%d\n",nlist,numedge, n,k);
  // For now, only support <= max int results
  FAISS_THROW_IF_NOT_FMT(n <=
                     (faiss::Index::idx_t) std::numeric_limits<int>::max(),
                     "GPU index only supports up to %zu indices",
                     (size_t) std::numeric_limits<int>::max());
  FAISS_THROW_IF_NOT_FMT(k <= 1024,
                     "GPU only supports k <= 1024 (requested %d)",
                     (int) k); // select limitation

  DeviceScope scope(device_);
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStream(device_);
  int maxThreads = getMaxThreadsCurrentDevice();
  constexpr int rowTileSize = 8;
  auto input = toDevice<float, 2>(resources_,
                                         device_,
                                          const_cast<float*>(x),
                                         stream,
                                         {(int) n, (int) dim});
      //  outputVec("input", input.data(),100,stream);

  auto edgedistinfoV = toDevice<float, 2>(resources_,
                                         device_,
                                         edgedistinfo,
                                         stream,
                                         {(int) nlist, (int) numedge});

  // Convert and copy int indices out
  auto edgeinfoV = toDevice<int, 2>(resources_,
                                                     device_,
                                                     edgeinfo,
                                                     stream,
                                                     {(int) nlist, (int) numedge});

      // Convert and copy int indices out
 auto assign1V = toDevice<int, 2>(resources_,
                                                     device_,
                                                     assign1,
                                                     stream,
                                                     {(int) n,(int)k});

       // Convert and copy int indices out
  auto assign2V = toDevice<int, 2>(resources_,
                                                     device_,
                                                     assign2,
                                                     stream,
                                                     {(int) n,(int)k});
            // Convert and copy int indices out
  auto lamdafV = toDevice<float, 2>(resources_,
                                                     device_,
                                                     lamdaf,
                                                     stream,
                                                     {(int) n,(int)k});

    if (input.template canCastResize<float4>()) {
        // Can load using the vectorized type
        auto inputV = input.template castResize<float4>();
        auto _cbV = data_->getVectorsFloat32Ref().template castResize<float4>();
        int dim = inputV.getSize(1);
        int numThreads = min(dim, maxThreads);

        auto grid = dim3(utils::divUp(inputV.getSize(0), rowTileSize));
        auto block = dim3(numThreads);

        uint shmSize=(1+4*numedge)*sizeof(float);
        auto smem = sizeof(float) * utils::divUp(numThreads, kWarpSize)+shmSize;
        printf("smem size: %d\n",smem);
       get1BinKernel_nms<float, float4, rowTileSize>
          <<<grid,block,smem,stream>>>(assign2V,lamdafV,assign1V,
         _cbV,inputV,edgedistinfoV,edgeinfoV);
        // Copy back if necessary
     //outputVecInt("assign2V", assign2V.data(),10,stream);

      fromDevice<int, 2>(assign2V, assign2, stream);
      fromDevice<float, 2>(lamdafV, lamdaf, stream);

    }



}

void
GpuIndexFlat::assignLambda(
      int n,
      float* lambdaf,
      uint8_t* lambda, float* lambdaInfo, int nlambda)const {
  if (n == 0) {
    return;
  }

  DeviceScope scope(device_);
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStream(device_);
  int maxThreads = getMaxThreadsCurrentDevice();
  constexpr int rowTileSize = 8;
  auto lambdafV = toDevice<float, 1>(resources_,
                                         device_,
                                         lambdaf,
                                         stream,
                                         {(int) n});
      //  outputVec("input", input.data(),100,stream);

  auto lambdaInfoV = toDevice<float, 1>(resources_,
                                         device_,
                                         lambdaInfo,
                                         stream,
                                         {(int) nlambda});


          // Convert and copy int indices out
  auto lambdaV = toDevice<uint8_t, 1>(resources_,
                                        device_,
                                        lambda,
                                        stream,
                                        {(int) n});


        auto grid = dim3(utils::divUp(lambdafV.getSize(0), rowTileSize));
        auto block = dim3(nlambda);

        uint shmSize=2*nlambda*sizeof(float);
        auto smem =shmSize;
        printf("smem size: %d,lambdafV.getSize(0):%d\n",smem,lambdafV.getSize(0));
       assignLambdaKernel<rowTileSize>
          <<<grid,block,smem,stream>>>(lambdafV,lambdaV,lambdaInfoV);

      fromDevice<uint8_t, 1>(lambdaV, lambda, stream);




}


void
GpuIndexFlat::assign1Base(
      Tensor<float,2,true> input,
      Tensor<int,2,true> assign1V,
      Tensor<int,2,true> assign2V,
      Tensor<float,2,true> lambdafV,
      Tensor<int,2,true> edgeinfoV,
      Tensor<float,2,true> edgedistinfoV,int k)const {
  if (input.getSize(0) == 0) {
    return;
  }
   printf("nlist : %d ,numedge : %d ,n  :%d,k:%d\n",edgeinfoV.getSize(0),edgeinfoV.getSize(1), input.getSize(0),k);
  // For now, only support <= max int results
  FAISS_THROW_IF_NOT_FMT(input.getSize(0) <=
                     (faiss::Index::idx_t) std::numeric_limits<int>::max(),
                     "GPU index only supports up to %zu indices",
                     (size_t) std::numeric_limits<int>::max());
  FAISS_THROW_IF_NOT_FMT(k <= 1024,
                     "GPU only supports k <= 1024 (requested %d)",
                     (int) k); // select limitation

  DeviceScope scope(device_);
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStream(device_);
  int maxThreads = getMaxThreadsCurrentDevice();
  constexpr int rowTileSize = 8;


    if (input.template canCastResize<float4>()) {
        // Can load using the vectorized type
        auto inputV = input.template castResize<float4>();
        auto _cbV = data_->getVectorsFloat32Ref().template castResize<float4>();
        int dim = inputV.getSize(1);
        int numThreads = min(dim, maxThreads);

        auto grid = dim3(utils::divUp(inputV.getSize(0), rowTileSize));
        auto block = dim3(numThreads);

        uint shmSize=(1+4*edgeinfoV.getSize(1))*sizeof(float);
        auto smem = sizeof(float) * utils::divUp(numThreads, kWarpSize)+shmSize;
        printf("smem size: %d\n",smem);
       get1BinKernel_nms<float, float4, rowTileSize>
          <<<grid,block,smem,stream>>>(assign2V,lambdafV,assign1V,
         _cbV,inputV,edgedistinfoV,edgeinfoV);
        // Copy back if necessary
     //outputVecInt("assign2V", assign2V.data(),10,stream);


    }



}



void
GpuIndexFlat::searchImpl_(faiss::Index::idx_t n,
                          const float* x,
                          faiss::Index::idx_t k,
                          float* distances,
                          faiss::Index::idx_t* labels) const {
  FAISS_ASSERT_MSG(false, "Should not be called");
}
void
GpuIndexFlat::searchNonPaged_(int n,
                              const float* x,
                              int k,
                              float* outDistancesData,
                              int* outIndicesData) const {
  Tensor<float, 2, true> outDistances(outDistancesData, {n, k});
  Tensor<int, 2, true> outIndices(outIndicesData, {n, k});

  auto stream = resources_->getDefaultStream(device_);

  // Make sure arguments are on the device we desire; use temporary
  // memory allocations to move it if necessary
  auto vecs = toDevice<float, 2>(resources_,
                                 device_,
                                 const_cast<float*>(x),
                                 stream,
                                 {n, (int) this->d});

  data_->query(vecs, k, outDistances, outIndices, true);
}

void
GpuIndexFlat::searchNonPagedWithGraph_(int n,
                              const float* x,
                              int k,
                              int nlist,
                              int edge,
                              int* graphIndiceData,
                              float* outDistancesData,
                              float* outGraphDistancesData,
                              int* outIndicesData) const {
  Tensor<float, 2, true> outDistances(outDistancesData, {n, k});
  Tensor<float, 2, true> outGraphDistances(outGraphDistancesData, {n, k*edge});
  Tensor<int, 2, true> graphIndices(graphIndiceData, {nlist, edge});

  Tensor<int, 2, true> outIndices(outIndicesData, {n, k});

  auto stream = resources_->getDefaultStream(device_);

  // Make sure arguments are on the device we desire; use temporary
  // memory allocations to move it if necessary
  Tensor<float, 2, true> vecs = toDevice<float, 2>(resources_,
                                 device_,
                                 const_cast<float*>(x),
                                 stream,
                                 {n, (int) this->d});

  data_->query(vecs, k,graphIndices,outDistances,outGraphDistances,outIndices, true);
}
void
GpuIndexFlat::buildGraphNonPaged_(int n,
                              int k,
                              float* outDistancesData,
                              int* outIndicesData) const {

   auto stream = resources_->getDefaultStream(device_);
  DeviceTensor<float, 2, true> outDistances(
    resources_->getMemoryManagerCurrentDevice(),
    {(int) n, (int) k+1}, stream);

  // FlatIndex only supports an interface returning int indices
  DeviceTensor<int, 2, true> outIndices(
    resources_->getMemoryManagerCurrentDevice(),
    {(int) n, (int) k+1}, stream);

  data_->query(data_->getVectorsFloat32Ref(), k+1, outDistances, outIndices, true);
  float * dis = outDistances.data();
  int*  indices = outIndices.data();
  for(int i=0;i<n;i++){
     fromDevice(dis+i*(k+1)+1, outDistancesData+i*k,k, stream);
     fromDevice(indices+i*(k+1)+1, outIndicesData+i*k,k, stream);
  }

}
void GpuIndexFlat::assignFlat (idx_t n, const float * x, int * labels, idx_t k)
{
  printf("Index::assignFlat,k: %d\n",k);
  float * distances = new float[n * k];
  ScopeDeleter<float> del(distances);
  searchInt (n, x, k, distances, labels);
}
void
GpuIndexFlat::searchFromCpuPaged_(int n,
                                  const float* x,
                                  int k,
                                  float* outDistancesData,
                                  int* outIndicesData) const {
  Tensor<float, 2, true> outDistances(outDistancesData, {n, k});
  Tensor<int, 2, true> outIndices(outIndicesData, {n, k});

  // Is pinned memory available?
  auto pinnedAlloc = resources_->getPinnedMemory();
  int pageSizeInVecs =
    (int) ((pinnedAlloc.second / 2) / (sizeof(float) * this->d));

  if (!pinnedAlloc.first || pageSizeInVecs < 1) {
    // Just page without overlapping copy with compute
    int batchSize = utils::nextHighestPowerOf2(
      (int) ((size_t) kNonPinnedPageSize /
             (sizeof(float) * this->d)));

    for (int cur = 0; cur < n; cur += batchSize) {
      int num = std::min(batchSize, n - cur);

      auto outDistancesSlice = outDistances.narrowOutermost(cur, num);
      auto outIndicesSlice = outIndices.narrowOutermost(cur, num);

      searchNonPaged_(num,
                      x + (size_t) cur * this->d,
                      k,
                      outDistancesSlice.data(),
                      outIndicesSlice.data());
    }

    return;
  }

  //
  // Pinned memory is available, so we can overlap copy with compute.
  // We use two pinned memory buffers, and triple-buffer the
  // procedure:
  //
  // 1 CPU copy -> pinned
  // 2 pinned copy -> GPU
  // 3 GPU compute
  //
  // 1 2 3 1 2 3 ...   (pinned buf A)
  //   1 2 3 1 2 ...   (pinned buf B)
  //     1 2 3 1 ...   (pinned buf A)
  // time ->
  //
  auto defaultStream = resources_->getDefaultStream(device_);
  auto copyStream = resources_->getAsyncCopyStream(device_);

  FAISS_ASSERT((size_t) pageSizeInVecs * this->d <=
               (size_t) std::numeric_limits<int>::max());

  float* bufPinnedA = (float*) pinnedAlloc.first;
  float* bufPinnedB = bufPinnedA + (size_t) pageSizeInVecs * this->d;
  float* bufPinned[2] = {bufPinnedA, bufPinnedB};

  // Reserve space on the GPU for the destination of the pinned buffer
  // copy
  DeviceTensor<float, 2, true> bufGpuA(
    resources_->getMemoryManagerCurrentDevice(),
    {(int) pageSizeInVecs, (int) this->d},
    defaultStream);
  DeviceTensor<float, 2, true> bufGpuB(
    resources_->getMemoryManagerCurrentDevice(),
    {(int) pageSizeInVecs, (int) this->d},
    defaultStream);
  DeviceTensor<float, 2, true>* bufGpus[2] = {&bufGpuA, &bufGpuB};

  // Copy completion events for the pinned buffers
  std::unique_ptr<CudaEvent> eventPinnedCopyDone[2];

  // Execute completion events for the GPU buffers
  std::unique_ptr<CudaEvent> eventGpuExecuteDone[2];

  // All offsets are in terms of number of vectors; they remain within
  // int bounds (as this function only handles max in vectors)

  // Current start offset for buffer 1
  int cur1 = 0;
  int cur1BufIndex = 0;

  // Current start offset for buffer 2
  int cur2 = -1;
  int cur2BufIndex = 0;

  // Current start offset for buffer 3
  int cur3 = -1;
  int cur3BufIndex = 0;

  while (cur3 < n) {
    // Start async pinned -> GPU copy first (buf 2)
    if (cur2 != -1 && cur2 < n) {
      // Copy pinned to GPU
      int numToCopy = std::min(pageSizeInVecs, n - cur2);

      // Make sure any previous execution has completed before continuing
      auto& eventPrev = eventGpuExecuteDone[cur2BufIndex];
      if (eventPrev.get()) {
        eventPrev->streamWaitOnEvent(copyStream);
      }

      CUDA_VERIFY(cudaMemcpyAsync(bufGpus[cur2BufIndex]->data(),
                                  bufPinned[cur2BufIndex],
                                  (size_t) numToCopy * this->d * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  copyStream));

      // Mark a completion event in this stream
      eventPinnedCopyDone[cur2BufIndex] =
        std::move(std::unique_ptr<CudaEvent>(new CudaEvent(copyStream)));

      // We pick up from here
      cur3 = cur2;
      cur2 += numToCopy;
      cur2BufIndex = (cur2BufIndex == 0) ? 1 : 0;
    }

    if (cur3 != -1 && cur3 < n) {
      // Process on GPU
      int numToProcess = std::min(pageSizeInVecs, n - cur3);

      // Make sure the previous copy has completed before continuing
      auto& eventPrev = eventPinnedCopyDone[cur3BufIndex];
      FAISS_ASSERT(eventPrev.get());

      eventPrev->streamWaitOnEvent(defaultStream);

      // Create tensor wrappers
      DeviceTensor<float, 2, true> input(bufGpus[cur3BufIndex]->data(),
                                         {numToProcess, this->d});
      auto outDistancesSlice = outDistances.narrowOutermost(cur3, numToProcess);
      auto outIndicesSlice = outIndices.narrowOutermost(cur3, numToProcess);

      data_->query(input, k,
                   outDistancesSlice,
                   outIndicesSlice, true);

      // Create completion event
      eventGpuExecuteDone[cur3BufIndex] =
        std::move(std::unique_ptr<CudaEvent>(new CudaEvent(defaultStream)));

      // We pick up from here
      cur3BufIndex = (cur3BufIndex == 0) ? 1 : 0;
      cur3 += numToProcess;
    }

    if (cur1 < n) {
      // Copy CPU mem to CPU pinned
      int numToCopy = std::min(pageSizeInVecs, n - cur1);

      // Make sure any previous copy has completed before continuing
      auto& eventPrev = eventPinnedCopyDone[cur1BufIndex];
      if (eventPrev.get()) {
        eventPrev->cpuWaitOnEvent();
      }

      memcpy(bufPinned[cur1BufIndex],
             x + (size_t) cur1 * this->d,
             (size_t) numToCopy * this->d * sizeof(float));

      // We pick up from here
      cur2 = cur1;
      cur1 += numToCopy;
      cur1BufIndex = (cur1BufIndex == 0) ? 1 : 0;
    }
  }
}

void
GpuIndexFlat::reconstruct(faiss::Index::idx_t key,
                          float* out) const {
  DeviceScope scope(device_);

  FAISS_THROW_IF_NOT_MSG(key < this->ntotal, "index out of bounds");
  auto stream = resources_->getDefaultStream(device_);

  if (config_.useFloat16) {
    auto vec = data_->getVectorsFloat32Copy(key, 1, stream);
    fromDevice(vec.data(), out, this->d, stream);
  } else {
    auto vec = data_->getVectorsFloat32Ref()[key];
    fromDevice(vec.data(), out, this->d, stream);
  }
}



template <int RowTileSize>
__global__ void calResidual(Tensor<float, 2, true> residualV,Tensor<int, 2, true> assign,
      Tensor<float, 2, true>  lamda,
       Tensor<float, 2, true> input,
       Tensor<float, 2, true> vec,
       Tensor<int, 2, true> edgeinfo) {
    //printf("sdfsadfsad\n");
    extern __shared__ char smemByte[];
    float* smem = (float*) smemByte;
    int dim = input.getSize(1);
    int d_edge = edgeinfo.getSize(1);
    int RowTileSize1= RowTileSize;
    bool lastRowTile = (blockIdx.x == (gridDim.x - 1));
    int rowStart = RowTileSize * blockIdx.x;

    if (lastRowTile) {
        RowTileSize1 = input.getSize(0) - rowStart;
    }
    // We are handling the very end of the input matrix rows
    for (int row = 0; row < RowTileSize1; ++row) {

        float val = input[rowStart + row][threadIdx.x];
        int id=assign[rowStart + row][0];
        int a = id/d_edge;
        int b = edgeinfo.data()[id];
        float la=lamda[rowStart + row][0];
        float vala = vec[a][threadIdx.x];
        float valb = vec[b][threadIdx.x];
        smem[row*dim+threadIdx.x]=val-((1-la)*vala+la*valb);

    }
     __syncthreads();
    for (int row = 0; row < RowTileSize1; ++row) {

        residualV[rowStart + row][threadIdx.x]=smem[row*dim+threadIdx.x];
    }
     __syncthreads();
}


void
GpuIndexFlat::compute_residual(faiss::Index::idx_t n,const float * x,
            float * residual,
            int* edgeInfo,
            float* lambda,
            int numedge,int nlist,
            int* assign) const{
  DeviceScope scope(device_);
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStream(device_);
  int maxThreads = getMaxThreadsCurrentDevice();
  constexpr int rowTileSize = 8;
   printf("d:%d \n",this->d);
  auto input = toDevice<float, 2>(resources_,
                                         device_,
                                          const_cast<float*>(x),
                                         stream,
                                         {(int) n, (int) this->d});

   auto residualV = toDevice<float, 2>(resources_,
                                         device_,
                                          residual,
                                         stream,
                                         {(int) n, (int) this->d});
  // Convert and copy int indices out
  auto edgeinfoV = toDevice<int, 2>(resources_,
                                                     device_,
                                                     edgeInfo,
                                                     stream,
                                                     {(int) nlist, (int) numedge});

      // Convert and copy int indices out
 auto  assignV = toDevice<int, 2>(resources_,
                                                     device_,
                                                     assign,
                                                     stream,
                                                     {(int) n,1});

          // Convert and copy int indices out
  auto lambdaV = toDevice<float, 2>(resources_,
                                                     device_,
                                                     lambda,
                                                     stream,
                                                     {(int) n,1});

        int dim = input.getSize(1);
        int numThreads = min(dim, maxThreads);

        auto grid = dim3(utils::divUp(input.getSize(0), rowTileSize));
        auto block = dim3(numThreads);

        auto smem = sizeof(float) *rowTileSize*dim;
        auto vec = data_->getVectorsFloat32Ref();
       calResidual<rowTileSize> <<<grid,block,smem,stream>>>(residualV,assignV,lambdaV, input,
       vec,edgeinfoV);
     // Copy back if necessary
    //outputVec("residualV",residualV.data(),128,stream);
      fromDevice<float, 2>(residualV, residual, stream);

}


void
GpuIndexFlat::compute_residual(faiss::Index::idx_t n,const float * x,
            float * residual,
            int* edgeInfo,
            uint8_t* lambda,
            float* lambdaInfo,
            int numedge,int nlist,
            int* assign) const{
  DeviceScope scope(device_);
  auto& mem = resources_->getMemoryManagerCurrentDevice();
  auto stream = resources_->getDefaultStream(device_);
  int maxThreads = getMaxThreadsCurrentDevice();
  constexpr int rowTileSize = 8;
   printf("d:%d \n",this->d);
  auto input = toDevice<float, 2>(resources_,
                                         device_,
                                          const_cast<float*>(x),
                                         stream,
                                         {(int) n, (int) this->d});

   auto residualV = toDevice<float, 2>(resources_,
                                         device_,
                                          residual,
                                         stream,
                                         {(int) n, (int) this->d});
  // Convert and copy int indices out
  auto edgeinfoV = toDevice<int, 2>(resources_,
                                                     device_,
                                                     edgeInfo,
                                                     stream,
                                                     {(int) nlist, (int) numedge});

      // Convert and copy int indices out
 auto  assignV = toDevice<int, 2>(resources_,
                                                     device_,
                                                     assign,
                                                     stream,
                                                     {(int) n,1});
    float* lambdaf = new float[n];
    for(int i=0;i<n;i++){
        lambdaf[i]=lambdaInfo[lambda[i]];
    }

          // Convert and copy int indices out
  auto lambdaV = toDevice<float, 2>(resources_,
                                                     device_,
                                                     lambdaf,
                                                     stream,
                                                     {(int) n,1});

        int dim = input.getSize(1);
        int numThreads = min(dim, maxThreads);

        auto grid = dim3(utils::divUp(input.getSize(0), rowTileSize));
        auto block = dim3(numThreads);

        auto smem = sizeof(float) *rowTileSize*dim;
        auto vec = data_->getVectorsFloat32Ref();
       calResidual<rowTileSize> <<<grid,block,smem,stream>>>(residualV,assignV,lambdaV, input,
       vec,edgeinfoV);
     // Copy back if necessary
    //outputVec("residualV",residualV.data(),128,stream);
      fromDevice<float, 2>(residualV, residual, stream);
        delete[] lambdaf;
}
void
GpuIndexFlat::reconstruct_n(faiss::Index::idx_t i0,
                            faiss::Index::idx_t num,
                            float* out) const {
  DeviceScope scope(device_);

  FAISS_THROW_IF_NOT_MSG(i0 < this->ntotal, "index out of bounds");
  FAISS_THROW_IF_NOT_MSG(i0 + num - 1 < this->ntotal, "num out of bounds");
  auto stream = resources_->getDefaultStream(device_);

  if (config_.useFloat16) {
    auto vec = data_->getVectorsFloat32Copy(i0, num, stream);
    fromDevice(vec.data(), out, num * this->d, stream);
  } else {
    auto vec = data_->getVectorsFloat32Ref()[i0];
    fromDevice(vec.data(), out, this->d * num, stream);
  }
}

void
GpuIndexFlat::verifySettings_() const {
  // If we want Hgemm, ensure that it is supported on this device
  if (config_.useFloat16Accumulator) {
#ifdef FAISS_USE_FLOAT16
    FAISS_THROW_IF_NOT_MSG(config_.useFloat16,
                       "useFloat16Accumulator can only be enabled "
                       "with useFloat16");

    FAISS_THROW_IF_NOT_FMT(getDeviceSupportsFloat16Math(config_.device),
                       "Device %d does not support Hgemm "
                       "(useFloat16Accumulator)",
                       config_.device);
#else
    FAISS_THROW_IF_NOT_MSG(false, "not compiled with float16 support");
#endif
  }
}

//
// GpuIndexFlatL2
//

GpuIndexFlatL2::GpuIndexFlatL2(GpuResources* resources,
                               faiss::IndexFlatL2* index,
                               GpuIndexFlatConfig config) :
    GpuIndexFlat(resources, index, config) {
}

GpuIndexFlatL2::GpuIndexFlatL2(GpuResources* resources,
                               int dims,
                               GpuIndexFlatConfig config) :
    GpuIndexFlat(resources, dims, faiss::METRIC_L2, config) {
}

void
GpuIndexFlatL2::copyFrom(faiss::IndexFlatL2* index) {
  GpuIndexFlat::copyFrom(index);
}

void
GpuIndexFlatL2::copyTo(faiss::IndexFlatL2* index) {
  GpuIndexFlat::copyTo(index);
}

//
// GpuIndexFlatIP
//

GpuIndexFlatIP::GpuIndexFlatIP(GpuResources* resources,
                               faiss::IndexFlatIP* index,
                               GpuIndexFlatConfig config) :
    GpuIndexFlat(resources, index, config) {
}

GpuIndexFlatIP::GpuIndexFlatIP(GpuResources* resources,
                               int dims,
                               GpuIndexFlatConfig config) :
    GpuIndexFlat(resources, dims, faiss::METRIC_INNER_PRODUCT, config) {
}

void
GpuIndexFlatIP::copyFrom(faiss::IndexFlatIP* index) {
  GpuIndexFlat::copyFrom(index);
}

void
GpuIndexFlatIP::copyTo(faiss::IndexFlatIP* index) {
  GpuIndexFlat::copyTo(index);
}



} } // namespace
