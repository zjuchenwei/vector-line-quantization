/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sys/time.h>


#include <iostream>
#include <fstream>
#include <stdexcept>



#include "../IndexPQ.h"
#include "../IndexIVFPQ.h"
#include "../IndexFlat.h"
#include "../index_io.h"


/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/


float * fvecs_read (const char *fname,
                    size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
    return (int*)fvecs_read(fname, d_out, n_out);
}


double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}
bool file_exists(const std::string& _name) {
	struct stat buffer;

	return (stat(_name.c_str(), &buffer) == 0);
}


void header(std::string fs, uint &num, uint &dim) {
     	std::ifstream fin(fs.c_str(), std::ios_base::in | std::ios_base::binary);
	if (!fin.good()) {
		fin.close();
		throw std::runtime_error("read error");
	}
	fin >> num;
	fin >> dim;
	fin.ignore();
	fin.close();
}

template<typename T>
void read(std::string fs, T *ptr, size_t len, size_t offset = 0) {
   	 std::ifstream fin(fs.c_str(), std::ios_base::in | std::ios_base::binary);
	if (!fin.good()) {
		fin.close();
		throw std::runtime_error("write error");
	}

	size_t num = 0;
	size_t dim = 0;

	fin >> num;
	fin >> dim;
	fin.ignore();
    std::cout << "num: " << num << "  dim: " << dim << std::endl;
	std::cout << "tellg: " << fin.tellg() << std::endl;
	std::cout << "offset: " << (sizeof(T) * offset) << " len: " << len << std::endl;

	fin.seekg(0, std::ios::beg);
	fin.seekg(20 + sizeof(T) * offset, std::ios::beg);
	std::cout << "tellg: " << fin.tellg() << std::endl;
	fin.read((char*) ptr, len*sizeof(T));
	fin.close();

}

float* readFloat(const char* _fn, size_t _dim, size_t _num, size_t _offset) {

	size_t offset = _offset * _dim;
	size_t length = _num * _dim;

	float * buf = new float[length];

	float *raw_data = new float[length];
	read<float>(_fn, raw_data, length, offset);

	for (int i = 0; i < length; i++)
		buf[i] = raw_data[i];

	delete[] raw_data;

	return buf;
}

float* readUint8(const char* _fn, size_t _dim, size_t _num, size_t _offset) {

	size_t offset = _offset * _dim;
	size_t length = _num * _dim;

	float * buf = new float[length];

	uint8_t *raw_data = new uint8_t[length];
	read<uint8_t>(_fn, raw_data, length, offset);
	for (int i = 0; i < length; i++)
		buf[i] = raw_data[i];

	delete[] raw_data;

	return buf;
}

int main ()
{
    double t0 = elapsed();

    // dimension of the vectors to index
    uint d;

    // size of the database we plan to index
    size_t nb;
    size_t add_bs = 1000*1000; // # size of the blocks to add

    // make a set of nt training vectors in the unit cube
    // (could be the database)
      uint nt;
    //float *xt = fvecs_read("/home/data/sift1m/learn.fvecs", &d, &nt);
    header("/home/data/sift1b/learn.umem", nt, d);
    const char *path_learn =
	"/home/data/sift1b/learn.umem";
    float *xt = readUint8(path_learn, d, 2000000, 0);
    //---------------------------------------------------------------
    // Define the core quantizer
    // We choose a multiple inverted index for faster training with less data
    // and because it usually offers best accuracy/speed trade-offs
    //
    // We here assume that its lifespan of this coarse quantizer will cover the
    // lifespan of the inverted-file quantizer IndexIVFFlat below
    // With dynamic allocation, one may give the responsability to free the
    // quantizer to the inverted-file index (with attribute do_delete_quantizer)
    //
    // Note: a regular clustering algorithm would be defined as:
    //       faiss::IndexFlatL2 coarse_quantizer (d);
    //
    // Use nhash=2 subquantizers used to define the product coarse quantizer
    // Number of bits: we will have 2^nbits_coarse centroids per subquantizer
    //                 meaning (2^12)^nhash distinct inverted lists
    //
    // The parameter bytes_per_code is determined by the memory
    // constraint, the dataset will use nb * (bytes_per_code + 8)
    // bytes.
    //
    // The parameter nbits_subq is determined by the size of the dataset to index.
    //
    size_t nhash = 2;
    size_t nbits_subq = 14;
    size_t ncentroids = 1 << (nhash * nbits_subq);  // total # of centroids
    int bytes_per_code = 16;
    std::string _name= "sift1b_"+std::to_string(nbits_subq)+"_" +std::to_string(bytes_per_code);
     faiss::IndexIVFPQ  *pIndex;
    faiss::MultiIndexQuantizer coarse_quantizer (d, nhash, nbits_subq);

    printf ("IMI (%ld,%ld): %ld virtual centroids (target: %ld base vectors)",
            nhash, nbits_subq, ncentroids, nb);

    // the coarse quantizer should not be dealloced before the index
    // 4 = nb of bytes per code (d must be a multiple of this)
    // 8 = nb of bits per sub-code (almost always 8)
    faiss::MetricType metric = faiss::METRIC_L2; // can be METRIC_INNER_PRODUCT
    faiss::IndexIVFPQ index (&coarse_quantizer, d, ncentroids, bytes_per_code, 8);
    index.quantizer_trains_alone = true;

    // define the number of probes. 2048 is for high-dim, overkill in practice
    // Use 4-1024 depending on the trade-off speed accuracy that you want
    index.nprobe = 256;

     std::string db_file = _name + "_trained_index.faissindex";
        if (!file_exists(db_file)) {
             // training.

            // The distribution of the training vectors should be the same
            // as the database vectors. It could be a sub-sample of the
            // database vectors, if sampling is not biased. Here we just
            // randomly generate the vectors.

            printf ("[%.3f s] Generating %ld vectors in %dD for training\n",
                    elapsed() - t0, nt, d);


            printf ("[%.3f s] Training the index\n", elapsed() - t0);
            index.verbose = true;
            index.train (2000000, xt);
            delete[] xt;
            faiss::write_index(&index, db_file.c_str());
        }
        pIndex = dynamic_cast<faiss::IndexIVFPQ *> (faiss::read_index (db_file.c_str()));

    // the index can be re-loaded later with
    // faiss::Index * idx = faiss::read_index("/tmp/trained_index.faissindex");

        db_file = _name +"_populated_index.faissindex";
        //db_file = "/tmp/populated_index.faissindex";
        if (!file_exists(db_file)) {
         // populating the database
          uint nb, d2;

        header("/home/data/sift1b/base.umem", nb, d2);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf ("[%.3f s] Indexing database, size %ld*%ld\n",
                elapsed() - t0, nb, d);
        printf ("[%.3f s] Adding the vectors to the index\n",
                elapsed() - t0);

        printf ("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);
         std::vector <long> ids (add_bs);
        for (size_t begin = 0; begin <nb; begin += add_bs) {
           size_t chunksize = std::min (add_bs, nb-begin);

           for (size_t i = 0; i < chunksize; i++) {
                ids[i] = begin+i;
            }
           float* xb = readUint8("/home/data/sift1b/base.umem", d2, chunksize, begin);
           pIndex->add_with_ids (chunksize,
                                xb,
                                ids.data());
            delete [] xb;
        }
        faiss::write_index(pIndex, db_file.c_str());

    }
          pIndex = dynamic_cast<faiss::IndexIVFPQ *> (faiss::read_index (db_file.c_str()));


    uint nq;
    float *xq;

    {
        printf ("[%.3f s] Loading queries\n", elapsed() - t0);

        uint d2;

       header("/home/data/sift1b/query.umem", nq, d2);
       xq = readUint8("/home/data/sift1b/query.umem", d2, nq,0);
       // xq = fvecs_read("/home/data/sift1m/query.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
        for (int v = 0; v < 128; v++) {
            std::cout << "\t" << xq[v];
            }
            std::cout << "read QuerySet " << xq[99] << std::endl;

    }
     size_t kgt; // nb of results per query in the GT
    faiss::Index::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors

    {
        printf ("[%.3f s] Loading ground truth for %ld queries\n",
                elapsed() - t0, nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int *gt_int = ivecs_read("/home/data/sift1b/gnd/idx_1000M.ivecs", &kgt, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::Index::idx_t[kgt * nq];
        for(int i = 0; i < kgt * nq; i++) {
            gt[i] = gt_int[i];
        }

          for (int v = 0; v < kgt ; v++) {
            std::cout << "\t" << gt[v];
            }
            std::cout << "read GT " <<gt[99] << std::endl;
        delete [] gt_int;
    }
    // A few notes on the internal format of the index:
    //
    // - the positing lists for PQ codes are index.codes, which is a
    //    std::vector < std::vector<uint8_t> >
    //   if n is the length of posting list #i, codes[i] has length bytes_per_code * n
    //
    // - the corresponding ids are stored in index.ids
    //
    // - given a vector float *x, finding which k centroids are
    //   closest to it (ie to find the nearest neighbors) can be done with
    //
    //   long *centroid_ids = new long[k];
    //   float *distances = new float[k];
    //   index.quantizer->search (1, x, k, dis, centroids_ids);
    //


    { // searching the database
        int k =128;
         double start = elapsed();
        printf ("[%.3f s] Searching the %d nearest neighbors "
                "of %ld vectors in the index\n",
                start - t0, k, nq);

        std::vector<faiss::Index::idx_t> nns (k * nq);
        std::vector<float>               dis (k * nq);
        pIndex->nprobe = 1024*2;
        pIndex->search (nq, xq, k, dis.data(), nns.data());
         double end = elapsed();
        printf ("[%.3f s] Query results (vector ids, then distances):\n",
                end - t0);

        for (int i = 0; i < 10; i++) {
            printf ("query %2d: ", i);
            for (int j = 0; j < 10; j++) {
                printf ("%7ld ", nns[j + i * k]);
            }
            printf ("\n     dis: ");
            for (int j = 0; j < 10; j++) {
                printf ("%7g ", dis[j + i * k]);
            }
            printf ("\n");
        }

            faiss::Index::idx_t *I = nns.data();
 //evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for(int i = 0; i < nq; i++) {
            int gt_nn = gt[i * kgt];
            for(int j = 0; j <k; j++) {
                if (I[i * k + j] == gt_nn) {
                    if(j < 1) n_1++;
                    if(j < 10) n_10++;
                    if(j < 100) n_100++;
                }
            }
        }
        printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        printf("R@100 = %.4f\n", n_100 / float(nq));
        std::cout << "avg. query time   " << (end-start)*1000 / static_cast<double>(nq) << "ms" << std::endl;
        std::cout << "total. query time " << (end-start) << "s" << std::endl;
    }
    delete [] xq;
    delete [] gt;
    return 0;
}
