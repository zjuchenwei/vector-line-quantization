/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "IndexScalarQuantizer.h"

#include <cstdio>
#include <algorithm>

#include <omp.h>

#include <immintrin.h>

#include "utils.h"

#include "FaissAssert.h"

namespace faiss {

/*******************************************************************
 * ScalarQuantizer implementation
 *
 * The main source of complexity is to support combinations of 4
 * variants without incurring runtime tests or virtual function calls:
 *
 * - 4 / 8 bits per code component
 * - uniform / non-uniform
 * - IP / L2 distance search
 * - scalar / AVX distance computation
 *
 * The appropriate Quantizer object is returned via select_quantizer
 * that hides the template mess.
 ********************************************************************/

#ifdef __AVX__
#define USE_AVX
#endif


namespace {

typedef Index::idx_t idx_t;
typedef ScalarQuantizer::QuantizerType QuantizerType;
typedef ScalarQuantizer::RangeStat RangeStat;


/*******************************************************************
 * Codec: converts between values in [0, 1] and an index in a code
 * array. The "i" parameter is the vector component index (not byte
 * index).
 */

struct Codec8bit {

    static void encode_component (float x, uint8_t *code, int i) {
        code[i] = (int)(255 * x);
    }

    static float decode_component (const uint8_t *code, int i) {
        return (code[i] + 0.5f) / 255.0f;
    }

#ifdef USE_AVX
    static __m256 decode_8_components (const uint8_t *code, int i) {
        uint64_t c8 = *(uint64_t*)(code + i);
        __m128i c4lo = _mm_cvtepu8_epi32 (_mm_set1_epi32(c8));
        __m128i c4hi = _mm_cvtepu8_epi32 (_mm_set1_epi32(c8 >> 32));
        // __m256i i8 = _mm256_set_m128i(c4lo, c4hi);
        __m256i i8 = _mm256_castsi128_si256 (c4lo);
        i8 = _mm256_insertf128_si256 (i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps (i8);
        __m256 half = _mm256_set1_ps (0.5f);
        f8 += half;
        __m256 one_255 = _mm256_set1_ps (1.f / 255.f);
        return f8 * one_255;
    }
#endif
};


struct Codec4bit {

    static void encode_component (float x, uint8_t *code, int i) {
        code [i / 2] |= (int)(x * 15.0) << ((i & 1) << 2);
    }

    static float decode_component (const uint8_t *code, int i) {
        return (((code[i / 2] >> ((i & 1) << 2)) & 0xf) + 0.5f) / 15.0f;
    }


#ifdef USE_AVX
    static __m256 decode_8_components (const uint8_t *code, int i) {
        uint32_t c4 = *(uint32_t*)(code + (i >> 1));
        uint32_t mask = 0x0f0f0f0f;
        uint32_t c4ev = c4 & mask;
        uint32_t c4od = (c4 >> 4) & mask;

        // the 8 lower bytes of c8 contain the values
        __m128i c8 = _mm_unpacklo_epi8 (_mm_set1_epi32(c4ev),
                                        _mm_set1_epi32(c4od));
        __m128i c4lo = _mm_cvtepu8_epi32 (c8);
        __m128i c4hi = _mm_cvtepu8_epi32 (_mm_srli_si128(c8, 4));
        __m256i i8 = _mm256_castsi128_si256 (c4lo);
        i8 = _mm256_insertf128_si256 (i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps (i8);
        __m256 half = _mm256_set1_ps (0.5f);
        f8 += half;
        __m256 one_255 = _mm256_set1_ps (1.f / 15.f);
        return f8 * one_255;
    }
#endif
};


/*******************************************************************
 * Similarity: gets vector components and computes a similarity wrt. a
 * query vector stored in the object
 */

struct SimilarityL2 {
    const float *y, *yi;
    explicit SimilarityL2 (const float * y): y(y) {}


    /******* scalar accumulator *******/

    float accu;

    void begin () {
        accu = 0;
        yi = y;
    }

    void add_component (float x) {
        float tmp = *yi++ - x;
        accu += tmp * tmp;
    }

    float result () {
        return accu;
    }

#ifdef USE_AVX
    /******* AVX accumulator *******/

    __m256 accu8;

    void begin_8 () {
        accu8 = _mm256_setzero_ps();
        yi = y;
    }

    void add_8_components (__m256 x) {
        __m256 yiv = _mm256_loadu_ps (yi);
        yi += 8;
        __m256 tmp = yiv - x;
        accu8 += tmp * tmp;
    }

    float result_8 () {
        __m256 sum = _mm256_hadd_ps(accu8, accu8);
        __m256 sum2 = _mm256_hadd_ps(sum, sum);
        // now add the 0th and 4th component
        return
            _mm_cvtss_f32 (_mm256_castps256_ps128(sum2)) +
            _mm_cvtss_f32 (_mm256_extractf128_ps(sum2, 1));
    }
#endif
};

struct SimilarityIP {
    const float *y, *yi;
    const float accu0;

    /******* scalar accumulator *******/

    float accu;

    SimilarityIP (const float * y, float accu0):
        y (y), accu0 (accu0) {}

    void begin () {
        accu = accu0;
        yi = y;
    }

    void add_component (float x) {
        accu +=  *yi++ * x;
    }

    float result () {
        return accu;
    }

#ifdef USE_AVX
    /******* AVX accumulator *******/

    __m256 accu8;

    void begin_8 () {
        accu8 = _mm256_setzero_ps();
        yi = y;
    }

    void add_8_components (__m256 x) {
        __m256 yiv = _mm256_loadu_ps (yi);
        yi += 8;
        accu8 += yiv * x;
    }

    float result_8 () {
        __m256 sum = _mm256_hadd_ps(accu8, accu8);
        __m256 sum2 = _mm256_hadd_ps(sum, sum);
        // now add the 0th and 4th component
        return
            accu0 +
            _mm_cvtss_f32 (_mm256_castps256_ps128(sum2)) +
            _mm_cvtss_f32 (_mm256_extractf128_ps(sum2, 1));
    }
#endif
};


/*******************************************************************
 * templatized distance functions
 */


template<class Quantizer, class Similarity>
float compute_distance(const Quantizer & quant, Similarity & sim,
                       const uint8_t *code)
{
    sim.begin();
    for (size_t i = 0; i < quant.d; i++) {
        float xi = quant.reconstruct_component (code, i);
        sim.add_component (xi);
    }
    return sim.result();
}

#ifdef USE_AVX
template<class Quantizer, class Similarity>
float compute_distance_8(const Quantizer & quant, Similarity & sim,
                         const uint8_t *code)
{
    sim.begin_8();
    for (size_t i = 0; i < quant.d; i += 8) {
        __m256 xi = quant.reconstruct_8_components (code, i);
        sim.add_8_components (xi);
    }
    return sim.result_8();
}
#endif


/*******************************************************************
 * Quantizer range training
 */

static float sqr (float x) {
    return x * x;
}


void train_Uniform(RangeStat rs, float rs_arg,
                   idx_t n, int k, const float *x,
                   std::vector<float> & trained)
{
    trained.resize (2);
    float & vmin = trained[0];
    float & vmax = trained[1];

    if (rs == ScalarQuantizer::RS_minmax) {
        vmin = HUGE_VAL; vmax = -HUGE_VAL;
        for (size_t i = 0; i < n; i++) {
            if (x[i] < vmin) vmin = x[i];
            if (x[i] > vmax) vmax = x[i];
        }
        float vexp = (vmax - vmin) * rs_arg;
        vmin -= vexp;
        vmax += vexp;
    } else if (rs == ScalarQuantizer::RS_meanstd) {
        double sum = 0, sum2 = 0;
        for (size_t i = 0; i < n; i++) {
            sum += x[i];
            sum2 += x[i] * x[i];
        }
        float mean = sum / n;
        float var = sum2 / n - mean * mean;
        float std = var <= 0 ? 1.0 : sqrt(var);

        vmin = mean - std * rs_arg ;
        vmax = mean + std * rs_arg ;
    } else if (rs == ScalarQuantizer::RS_quantiles) {
        std::vector<float> x_copy(n);
        memcpy(x_copy.data(), x, n * sizeof(*x));
        // TODO just do a qucikselect
        std::sort(x_copy.begin(), x_copy.end());
        int o = int(rs_arg * n);
        if (o < 0) o = 0;
        if (o > n - o) o = n / 2;
        vmin = x_copy[o];
        vmax = x_copy[n - 1 - o];

    } else if (rs == ScalarQuantizer::RS_optim) {
        float a, b;
        float sx = 0;
        {
            vmin = HUGE_VAL, vmax = -HUGE_VAL;
            for (size_t i = 0; i < n; i++) {
                if (x[i] < vmin) vmin = x[i];
                if (x[i] > vmax) vmax = x[i];
                sx += x[i];
            }
            b = vmin;
            a = (vmax - vmin) / (k - 1);
        }
        int verbose = false;
        int niter = 2000;
        float last_err = -1;
        int iter_last_err = 0;
        for (int it = 0; it < niter; it++) {
            float sn = 0, sn2 = 0, sxn = 0, err1 = 0;

            for (idx_t i = 0; i < n; i++) {
                float xi = x[i];
                float ni = floor ((xi - b) / a + 0.5);
                if (ni < 0) ni = 0;
                if (ni >= k) ni = k - 1;
                err1 += sqr (xi - (ni * a + b));
                sn  += ni;
                sn2 += ni * ni;
                sxn += ni * xi;
            }

            if (err1 == last_err) {
                iter_last_err ++;
                if (iter_last_err == 16) break;
            } else {
                last_err = err1;
                iter_last_err = 0;
            }

            float det = sqr (sn) - sn2 * n;

            b = (sn * sxn - sn2 * sx) / det;
            a = (sn * sx - n * sxn) / det;
            if (verbose) {
                printf ("it %d, err1=%g            \r", it, err1);
                fflush(stdout);
            }
        }
        if (verbose) printf("\n");

        vmin = b;
        vmax = b + a * (k - 1);

    } else {
        FAISS_THROW_MSG ("Invalid qtype");
    }
    vmax -= vmin;
}

void train_NonUniform(RangeStat rs, float rs_arg,
                      idx_t n, int d, int k, const float *x,
                      std::vector<float> & trained)
{
    trained.resize (2 * d);
    float * vmin = trained.data();
    float * vmax = trained.data() + d;
    if (rs == ScalarQuantizer::RS_minmax) {
        memcpy (vmin, x, sizeof(*x) * d);
        memcpy (vmax, x, sizeof(*x) * d);
        for (size_t i = 1; i < n; i++) {
            const float *xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                if (xi[j] < vmin[j]) vmin[j] = xi[j];
                if (xi[j] > vmax[j]) vmax[j] = xi[j];
            }
        }
        float *vdiff = vmax;
        for (size_t j = 0; j < d; j++) {
            float vexp = (vmax[j] - vmin[j]) * rs_arg;
            vmin[j] -= vexp;
            vmax[j] += vexp;
            vdiff [j] = vmax[j] - vmin[j];
        }
    } else {
        // transpose
        std::vector<float> xt(n * d);
        for (size_t i = 1; i < n; i++) {
            const float *xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                xt[j * n + i] = xi[j];
            }
        }
        std::vector<float> trained_d(2);
#pragma omp parallel for
        for (size_t j = 0; j < d; j++) {
            train_Uniform(rs, rs_arg,
                          n, k, xt.data() + j * n,
                          trained_d);
            vmin[j] = trained_d[0];
            vmax[j] = trained_d[1];
        }
    }
}


/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 */



struct Quantizer {
    virtual void encode_vector(const float *x, uint8_t *code) const = 0;
    virtual void decode_vector(const uint8_t *code, float *x) const = 0;

    virtual float compute_distance_L2 (SimilarityL2 &sim,
                                       const uint8_t * codes) const = 0;
    virtual float compute_distance_IP (SimilarityIP &sim,
                                       const uint8_t * codes) const = 0;

    virtual ~Quantizer() {}
};




template<class Codec>
struct QuantizerUniform: Quantizer {
    const size_t d;
    const float vmin, vdiff;

    QuantizerUniform(size_t d, const std::vector<float> &trained):
        d(d), vmin(trained[0]), vdiff(trained[1]) {
    }

    void encode_vector(const float* x, uint8_t* code) const override {
      for (size_t i = 0; i < d; i++) {
        float xi = (x[i] - vmin) / vdiff;
        if (xi < 0)
          xi = 0;
        if (xi > 1.0)
          xi = 1.0;
        Codec::encode_component(xi, code, i);
      }
    }

    void decode_vector(const uint8_t* code, float* x) const override {
      for (size_t i = 0; i < d; i++) {
        float xi = Codec::decode_component(code, i);
        x[i] = vmin + xi * vdiff;
      }
    }

    float reconstruct_component (const uint8_t * code, int i) const
    {
        float xi = Codec::decode_component (code, i);
        return vmin + xi * vdiff;
    }

#ifdef USE_AVX
    __m256 reconstruct_8_components (const uint8_t * code, int i) const
    {
        __m256 xi = Codec::decode_8_components (code, i);
        return _mm256_set1_ps(vmin) + xi * _mm256_set1_ps (vdiff);
    }
#endif

    float compute_distance_L2(SimilarityL2& sim, const uint8_t* codes)
        const override {
      return compute_distance(*this, sim, codes);
    }

    float compute_distance_IP(SimilarityIP& sim, const uint8_t* codes)
        const override {
      return compute_distance(*this, sim, codes);
    }
};

#ifdef USE_AVX
template<class Codec>
struct QuantizerUniform8: QuantizerUniform<Codec> {

    QuantizerUniform8 (size_t d, const std::vector<float> &trained):
        QuantizerUniform<Codec> (d, trained) {}

    float compute_distance_L2(SimilarityL2& sim, const uint8_t* codes)
        const override {
      return compute_distance_8(*this, sim, codes);
    }

    float compute_distance_IP(SimilarityIP& sim, const uint8_t* codes)
        const override {
      return compute_distance_8(*this, sim, codes);
    }
};
#endif





template<class Codec>
struct QuantizerNonUniform: Quantizer {
    const size_t d;
    const float *vmin, *vdiff;

    QuantizerNonUniform(size_t d, const std::vector<float> &trained):
        d(d), vmin(trained.data()), vdiff(trained.data() + d) {}

    void encode_vector(const float* x, uint8_t* code) const override {
      for (size_t i = 0; i < d; i++) {
        float xi = (x[i] - vmin[i]) / vdiff[i];
        if (xi < 0)
          xi = 0;
        if (xi > 1.0)
          xi = 1.0;
        Codec::encode_component(xi, code, i);
      }
    }

    void decode_vector(const uint8_t* code, float* x) const override {
      for (size_t i = 0; i < d; i++) {
        float xi = Codec::decode_component(code, i);
        x[i] = vmin[i] + xi * vdiff[i];
      }
    }

    float reconstruct_component (const uint8_t * code, int i) const
    {
        float xi = Codec::decode_component (code, i);
        return vmin[i] + xi * vdiff[i];
    }

#ifdef USE_AVX
    __m256 reconstruct_8_components (const uint8_t * code, int i) const
    {
        __m256 xi = Codec::decode_8_components (code, i);
        return _mm256_loadu_ps(vmin + i) + xi * _mm256_loadu_ps (vdiff + i);
    }
#endif

    float compute_distance_L2(SimilarityL2& sim, const uint8_t* codes)
        const override {
      return compute_distance(*this, sim, codes);
    }

    float compute_distance_IP(SimilarityIP& sim, const uint8_t* codes)
        const override {
      return compute_distance(*this, sim, codes);
    }
};

#ifdef USE_AVX
template<class Codec>
struct QuantizerNonUniform8: QuantizerNonUniform<Codec> {

    QuantizerNonUniform8 (size_t d, const std::vector<float> &trained):
        QuantizerNonUniform<Codec> (d, trained) {}

    float compute_distance_L2(SimilarityL2& sim, const uint8_t* codes)
        const override {
      return compute_distance_8(*this, sim, codes);
    }

    float compute_distance_IP(SimilarityIP& sim, const uint8_t* codes)
        const override {
      return compute_distance_8(*this, sim, codes);
    }
};
#endif





Quantizer *select_quantizer (
       QuantizerType qtype,
       size_t d, const std::vector<float> & trained)
{
#ifdef USE_AVX
    if (d % 8 == 0) {
        switch(qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerNonUniform8<Codec8bit>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerNonUniform8<Codec4bit>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerUniform8<Codec8bit>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerUniform8<Codec4bit>(d, trained);
        }
    } else
#endif
    {
        switch(qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerNonUniform<Codec8bit>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerNonUniform<Codec4bit>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerUniform<Codec8bit>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerUniform<Codec4bit>(d, trained);
        }
    }
    FAISS_THROW_MSG ("unknown qtype");
    return nullptr;
}

Quantizer *select_quantizer (const ScalarQuantizer &sq)
{
    return select_quantizer (sq.qtype, sq.d, sq.trained);
}


} // anonymous namespace



/*******************************************************************
 * ScalarQuantizer implementation
 ********************************************************************/

ScalarQuantizer::ScalarQuantizer
          (size_t d, QuantizerType qtype):
              qtype (qtype), rangestat(RS_minmax), rangestat_arg(0), d (d)
{
    switch (qtype) {
    case QT_8bit: case QT_8bit_uniform:
        code_size = d;
        break;
    case QT_4bit: case QT_4bit_uniform:
        code_size = (d + 1) / 2;
        break;
    }

}

ScalarQuantizer::ScalarQuantizer ():
    qtype(QT_8bit),
    rangestat(RS_minmax), rangestat_arg(0), d (0), code_size(0)
{}

void ScalarQuantizer::train (size_t n, const float *x)
{
    int bit_per_dim =
        qtype == QT_4bit_uniform ? 4 :
        qtype == QT_4bit ? 4 :
        qtype == QT_8bit_uniform ? 8 :
        qtype == QT_8bit ? 8 : -1;

    switch (qtype) {
    case QT_4bit_uniform: case QT_8bit_uniform:
        train_Uniform (rangestat, rangestat_arg,
                       n * d, 1 << bit_per_dim, x, trained);
        break;
    case QT_4bit: case QT_8bit:
        train_NonUniform (rangestat, rangestat_arg,
                          n, d, 1 << bit_per_dim, x, trained);
        break;
    }
}

void ScalarQuantizer::compute_codes (const float * x,
                                     uint8_t * codes,
                                     size_t n) const
{
    Quantizer *squant = select_quantizer (*this);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        squant->encode_vector (x + i * d, codes + i * code_size);
    delete squant;
}

void ScalarQuantizer::decode (const uint8_t *codes, float *x, size_t n) const
{
    Quantizer *squant = select_quantizer (*this);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        squant->decode_vector (codes + i * code_size, x + i * d);
    delete squant;
}

/*******************************************************************
 * IndexScalarQuantizer implementation
 ********************************************************************/

IndexScalarQuantizer::IndexScalarQuantizer
                      (int d, ScalarQuantizer::QuantizerType qtype,
                       MetricType metric):
          Index(d, metric),
          sq (d, qtype)
{
    is_trained = false;
    code_size = sq.code_size;
}


IndexScalarQuantizer::IndexScalarQuantizer ():
    IndexScalarQuantizer(0, ScalarQuantizer::QT_8bit)
{}

void IndexScalarQuantizer::train(idx_t n, const float* x)
{
    sq.train(n, x);
    is_trained = true;
}

void IndexScalarQuantizer::add(idx_t n, const float* x)
{
    FAISS_THROW_IF_NOT (is_trained);
    codes.resize ((n + ntotal) * code_size);
    sq.compute_codes (x, &codes[ntotal * code_size], n);
    ntotal += n;
}

void IndexScalarQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const
{
    Quantizer *squant = select_quantizer (sq);
    ScopeDeleter1<Quantizer> del(squant);
    FAISS_THROW_IF_NOT (is_trained);

    if (metric_type == METRIC_INNER_PRODUCT) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            idx_t *idxi = labels + i * k;
            float *simi = distances + i * k;
            minheap_heapify (k, simi, idxi);

            SimilarityIP sim(x + i * d, 0);
            const uint8_t *ci = codes.data ();

            for (size_t j = 0; j < ntotal; j++) {
                float accu = squant->compute_distance_IP(sim, ci);

                if (accu > simi [0]) {
                    minheap_pop (k, simi, idxi);
                    minheap_push (k, simi, idxi, accu, j);
                }
                ci += code_size;
            }
        }
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            idx_t *idxi = labels + i * k;
            float *simi = distances + i * k;
            maxheap_heapify (k, simi, idxi);

            SimilarityL2 sim(x + i * d);
            const uint8_t *ci = codes.data ();

            for (size_t j = 0; j < ntotal; j++) {
                float accu = squant->compute_distance_L2(sim, ci);

                if (accu < simi [0]) {
                    maxheap_pop (k, simi, idxi);
                    maxheap_push (k, simi, idxi, accu, j);
                }
                ci += code_size;
            }

        }
    }

}

void IndexScalarQuantizer::reset()
{
    codes.clear();
    ntotal = 0;
}

void IndexScalarQuantizer::reconstruct_n(
             idx_t i0, idx_t ni, float* recons) const
{
    Quantizer *squant = select_quantizer (sq);
    ScopeDeleter1<Quantizer> del (squant);
    for (size_t i = 0; i < ni; i++) {
        squant->decode_vector(&codes[(i + i0) * code_size], recons + i * d);
    }
}

void IndexScalarQuantizer::reconstruct(idx_t key, float* recons) const
{
    reconstruct_n(key, 1, recons);
}


/*******************************************************************
 * IndexIVFScalarQuantizer implementation
 ********************************************************************/

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer
          (Index *quantizer, size_t d, size_t nlist,
           QuantizerType qtype, MetricType metric):
              IndexIVF (quantizer, d, nlist, metric),
              sq (d, qtype)
{
    code_size = sq.code_size;
    is_trained = false;
    codes.resize(nlist);
}

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer ():
      IndexIVF (), code_size (0)
{}

void IndexIVFScalarQuantizer::train_residual (idx_t n, const float *x)
{
    long * idx = new long [n];
    ScopeDeleter<long> del (idx);
    quantizer->assign (n, x, idx);
    float *residuals = new float [n * d];
    ScopeDeleter<float> del2 (residuals);

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        quantizer->compute_residual (x + i * d, residuals + i * d, idx[i]);
    }

    sq.train (n, residuals);

}


void IndexIVFScalarQuantizer::add_with_ids
       (idx_t n, const float * x, const long *xids)
{
    FAISS_THROW_IF_NOT (is_trained);
    long * idx = new long [n];
    ScopeDeleter<long> del (idx);
    quantizer->assign (n, x, idx);
    size_t nadd = 0;
    Quantizer *squant = select_quantizer (sq);
    ScopeDeleter1<Quantizer> del2 (squant);

#pragma omp parallel reduction(+: nadd)
    {
        std::vector<float> residual (d);
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        for (size_t i = 0; i < n; i++) {

            long list_no = idx [i];
            if (list_no >= 0 && list_no % nt == rank) {
                long id = xids ? xids[i] : ntotal + i;

                assert (list_no < nlist);

                ids[list_no].push_back (id);
                nadd++;
                quantizer->compute_residual (
                      x + i * d, residual.data(), list_no);

                size_t cur_size = codes[list_no].size();
                codes[list_no].resize (cur_size + code_size);

                squant->encode_vector (residual.data(),
                                       codes[list_no].data() + cur_size);
            }
        }
    }
    ntotal += nadd;
}


void search_with_probes_ip (const IndexIVFScalarQuantizer & index,
                            const float *x,
                            const idx_t *cent_ids, const float *cent_dis,
                            const Quantizer & quant,
                            int k, float *simi, idx_t *idxi)
{
    int nprobe = index.nprobe;
    size_t code_size = index.code_size;
    size_t d = index.d;
    std::vector<float> decoded(d);
    minheap_heapify (k, simi, idxi);
    for (int i = 0; i < nprobe; i++) {
        idx_t list_no = cent_ids[i];
        if (list_no < 0) break;
        float accu0 = cent_dis[i];

        const std::vector<idx_t> & ids = index.ids[list_no];
        const uint8_t* codes = index.codes[list_no].data();

        SimilarityIP sim(x, accu0);

        for (size_t j = 0; j < ids.size(); j++) {

            float accu = quant.compute_distance_IP(sim, codes);

            if (accu > simi [0]) {
                minheap_pop (k, simi, idxi);
                minheap_push (k, simi, idxi, accu, ids[j]);
            }
            codes += code_size;
        }

    }
    minheap_reorder (k, simi, idxi);
}

void search_with_probes_L2 (const IndexIVFScalarQuantizer & index,
                            const float *x_in,
                            const idx_t *cent_ids,
                            const Index *quantizer,
                            const Quantizer & quant,
                            int k, float *simi, idx_t *idxi)
{
    int nprobe = index.nprobe;
    size_t code_size = index.code_size;
    size_t d = index.d;
    std::vector<float> decoded(d), x(d);
    maxheap_heapify (k, simi, idxi);
    for (int i = 0; i < nprobe; i++) {
        idx_t list_no = cent_ids[i];
        if (list_no < 0) break;

        const std::vector<idx_t> & ids = index.ids[list_no];
        const uint8_t* codes = index.codes[list_no].data();

        // shift of x_in wrt centroid
        quantizer->compute_residual (x_in, x.data(), list_no);

        SimilarityL2 sim(x.data());

        for (size_t j = 0; j < ids.size(); j++) {

            float dis = quant.compute_distance_L2 (sim, codes);

            if (dis < simi [0]) {
                maxheap_pop (k, simi, idxi);
                maxheap_push (k, simi, idxi, dis, ids[j]);
            }
            codes += code_size;
        }
    }
    maxheap_reorder (k, simi, idxi);
}


void IndexIVFScalarQuantizer::search (idx_t n, const float *x, idx_t k,
                                      float *distances, idx_t *labels) const
{
    FAISS_THROW_IF_NOT (is_trained);
    idx_t *idx = new idx_t [n * nprobe];
    ScopeDeleter<idx_t> del (idx);
    float *dis = new float [n * nprobe];
    ScopeDeleter<float> del2 (dis);

    quantizer->search (n, x, nprobe, dis, idx);

    Quantizer *squant = select_quantizer (sq);
    ScopeDeleter1<Quantizer> del3(squant);

    if (metric_type == METRIC_INNER_PRODUCT) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            search_with_probes_ip (*this, x + i * d,
                                   idx + i * nprobe, dis + i * nprobe, *squant,
                                   k, distances + i * k, labels + i * k);
        }
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            search_with_probes_L2 (*this, x + i * d,
                                   idx + i * nprobe, quantizer, *squant,
                                   k, distances + i * k, labels + i * k);
        }
    }

}


void IndexIVFScalarQuantizer::merge_from_residuals (IndexIVF & other_in) {
    IndexIVFScalarQuantizer &other =
        dynamic_cast<IndexIVFScalarQuantizer &> (other_in);
    for (int i = 0; i < nlist; i++) {
        std::vector<uint8_t> & src = other.codes[i];
        std::vector<uint8_t> & dest = codes[i];
        dest.insert (dest.end(), src.begin (), src.end ());
        src.clear ();
    }

}


}
