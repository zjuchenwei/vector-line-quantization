/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "GpuIndexIVF.h"
#include "utils/Tensor.cuh"
#include <vector>

namespace faiss { struct IndexIVFPQ; }

namespace faiss { namespace gpu {

class GpuIndexFlat;
class IVFPQ;

struct GpuIndexIVFPQConfig : public GpuIndexIVFConfig {
  inline GpuIndexIVFPQConfig()
      : useFloat16LookupTables(false),
        usePrecomputedTables(false) {
  }

  /// Whether or not float16 residual distance tables are used in the
  /// list scanning kernels. When subQuantizers * 2^bitsPerCode >
  /// 16384, this is required.
  bool useFloat16LookupTables;

  /// Whether or not we enable the precomputed table option for
  /// search, which can substantially increase the memory requirement.
  bool usePrecomputedTables;
};

/// IVFPQ index for the GPU
class GpuIndexIVFPQ : public GpuIndexIVF {
 public:
  /// Construct from a pre-existing faiss::IndexIVFPQ instance, copying
  /// data over to the given GPU, if the input index is trained.
  GpuIndexIVFPQ(GpuResources* resources,
                const faiss::IndexIVFPQ* index,
                GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig());

  /// Construct an empty index
  GpuIndexIVFPQ(GpuResources* resources,
                int dims,
                int nlist,
                int subQuantizers,
                int bitsPerCode,
                faiss::MetricType metric,
                GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig());

 /// Construct an empty index
  GpuIndexIVFPQ(GpuResources* resources,
                int dims,
                int nlist,
                int subQuantizers,
                int bitsPerCode,
                int nedge,
                int nLambda,
                faiss::MetricType metric,
                GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig());

  ~GpuIndexIVFPQ() override;

  int nLambda_;

  int numedge_;
  int begin_;
  int end_;
  int w1_;

  int* edgeInfo_;
  float* edgeDistInfo_;
  float* lambdaInfo_;
  float* constInfo_;
  /// Reserve space on the GPU for the inverted lists for `num`
  /// vectors, assumed equally distributed among

  /// Initialize ourselves from the given CPU index; will overwrite
  /// all data in ourselves
  void copyFrom(const faiss::IndexIVFPQ* index);

  /// Copy ourselves to the given CPU index; will overwrite all data
  /// in the index instance
  void copyTo(faiss::IndexIVFPQ* index) const;

  /// Reserve GPU memory in our inverted lists for this number of vectors
  void reserveMemory(size_t numVecs);

  /// Enable or disable pre-computed codes
  void setPrecomputedCodes(bool enable);

  /// Are pre-computed codes enabled?
  bool getPrecomputedCodes() const;

  /// Return the number of sub-quantizers we are using
  int getNumSubQuantizers() const;

  /// Return the number of bits per PQ code
  int getBitsPerCode() const;

  /// Return the number of centroids per PQ code (2^bits per code)
  int getCentroidsPerSubQuantizer() const;

  /// After adding vectors, one can call this to reclaim device memory
  /// to exactly the amount needed. Returns space reclaimed in bytes
  size_t reclaimMemory();

  /// Clears out all inverted lists, but retains the coarse and
  /// product centroid information
  void reset() override;

  void train(Index::idx_t n, const float* x) override;

  void readTreeFromFile(Index::idx_t n, const float* x,const std::string& _name);

  /// For debugging purposes, return the list length of a particular
  /// list
  int getListLength(int listId) const;

  /// For debugging purposes, return the list codes of a particular
  /// list
  std::vector<unsigned char> getListCodes(int listId) const;
   std::vector<unsigned char> getListLambdas(int listId)const;

  /// For debugging purposes, return the list indices of a particular
  /// list
  std::vector<long> getListIndices(int listId) const;
  void merge(faiss::Index::idx_t* nns,
                           float* dist,
                           int k,
                           int nq,
                           int nprocess,
                           float* distances,
                           faiss::Index::idx_t* labels) const;

  void writeCodebookToFile(const std::string& _name);
  void writeDbToFile(const std::string& _name);

  void readCodebookFromFile(const std::string& _name);
  void readDbFromFile(const std::string& _name);
  void readDbFromFile(const std::string& _name,size_t nb);
  void readDbFromFile1(const std::string& _name);
  void readDbFromFile(const std::string& _name,int pronum,int rank);
  void readDbFromFile(const std::string& _name,size_t nb,int pronum,int rank);
  void writeCentroidsToFile(const std::string& _name);
  void buildGraph_(faiss::Index::idx_t n, const float* x);
  void testDistance_(faiss::Index::idx_t n,
                           const float* x,
                           faiss::Index::idx_t k,
                           float* distances,
                           faiss::Index::idx_t* labels) const;
 int* add_with_ids1(Index::idx_t n,const float* x,const Index::idx_t* ids);
 void generateNNs(Tensor<float, 2, true>& vecs,Tensor<float, 2, true>& vecsq,
                             Tensor<long, 1, true>& indices, Tensor<long, 2, true>& nns,Tensor<float, 2, true>& dists);
 void add_with_ids2(Index::idx_t n,Index::idx_t nq, uint kgt,
                       const float* x,
                       const float* xq,
                       const Index::idx_t* ids,Index::idx_t* nns,float* dists);
 void search1(
      faiss::Index::idx_t n,
      const float* x,
      faiss::Index::idx_t k,
      float* distances,
      faiss::Index::idx_t* labels) const;
 protected:
  /// Called from GpuIndex for add/add_with_ids
  void addImpl_(
      faiss::Index::idx_t n,
      const float* x,
      const faiss::Index::idx_t* ids) override;

  /// Called from GpuIndex for search
  void searchImpl_(
      faiss::Index::idx_t n,
      const float* x,
      faiss::Index::idx_t k,
      float* distances,
      faiss::Index::idx_t* labels) const override;

        /// Called from GpuIndex for search1
  void searchImpl1_(
      faiss::Index::idx_t n,
      const float* x,
      faiss::Index::idx_t k,
      float* distances,
      faiss::Index::idx_t* labels) const;

 private:
  void verifySettings_() const;

  void trainResidualQuantizer_(Index::idx_t n, const float* x);

  int classifyAndAddVectors(Tensor<float, 2, true>& vecs,Tensor<long, 1, true>& indices);
  void classifyAndAddVectors1(Tensor<float, 2, true>& vecs,Tensor<long, 1, true>& indices ,int* assign1);
  void addImpl1_(Index::idx_t n,
                        const float* x,
                        const Index::idx_t* xids,int* assign1);

   void  calConstQ (float* constq,
                Tensor<int, 2, true>& edgeinfo,
                Tensor<float, 2, true>& edgeDistinfo,
                 Tensor<int, 2, true>& listIds2d,
                Tensor<int, 2, true>& encodings,
                 Tensor<float, 1, true> lambInfo,
               Tensor<uint8_t, 2, true> lambda,
                Tensor<half, 3, true>& precomputedCode,
                Tensor<float, 1, true>& subQuantizerNorms);


 private:
  GpuIndexIVFPQConfig ivfpqConfig_;

  /// Number of sub-quantizers per encoded vector
  int subQuantizers_;

  /// Bits per sub-quantizer code
  int bitsPerCode_;

  /// Desired inverted list memory reservation
  size_t reserveMemoryVecs_;


  /// The product quantizer instance that we own; contains the
  /// inverted lists
  IVFPQ* index_;
   float* pqData_;
};

} } // namespace
