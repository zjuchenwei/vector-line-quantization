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


#include "../StandardGpuResources.h"
#include "../GpuIndexIVFPQ.h"

#include "../GpuAutoTune.h"
#include "../../index_io.h"
#include <iostream>
#include <fstream>
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
	     std::ifstream fin(fs.c_str(), std::ofstream::in | std::ofstream::binary);
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
     std::ifstream fin(fs.c_str(), std::ofstream::in | std::ofstream::binary);
	if (!fin.good()) {
		fin.close();
		throw std::runtime_error("write error");
	}

	size_t num = 0;
	size_t dim = 0;

	fin >> num;
	fin >> dim;
	fin.ignore();

	std::cout << "tellg: " << fin.tellg() << std::endl;
	std::cout << "offset: " << (sizeof(T) * offset) << " len: " << len << std::endl;

	fin.seekg(0, std::ios::beg);
	fin.seekg(20 + sizeof(T) * offset, std::ios::beg);
	fin.read((char*) ptr, (len) * sizeof(T));
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
    uint d ;
     int nlist=32*1024;
     int nsub = 8;
     int nedge = 64;

     //int tempmem = 780*1024*1024;

     size_t tempmem = 2000*1024*1024;

     std::cout << "tempmem " << tempmem << std::endl;
     std::string _name="sift1b";

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    uint nb;
    header("/home/data/sift1b/base.umem", nb, d);
   // float *xt = fvecs_read("/home/data/sift1m/learn.fvecs", &d, &nt);
    int dev_no = 0;
    /*
    printf ("[%.3f s] Begin d=%d nb=%ld nt=%nt dev_no=%d\n",
            elapsed() - t0, d, nb, nt, dev_no);
    */
    // a reasonable number of centroids to index nb vectors

    faiss::gpu::StandardGpuResources resources;
    resources.setTempMemory(tempmem);


    // the coarse quantizer should not be dealloced before the index
    // 4 = nb of bytes per code (d must be a multiple of this)
    // 8 = nb of bits per sub-code (almost always 8)

  //INDICES_CPU = 0,
  //INDICES_IVF = 1,
  //INDICES_32_BIT = 2,
  //INDICES_64_BIT = 3,
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = dev_no;
    faiss::gpu::GpuClonerOptions co;
    co.useFloat16 = false;
    config.indicesOptions = faiss::gpu::INDICES_CPU;
    config.flatConfig.useFloat16 = co.useFloat16CoarseQuantizer;
    config.flatConfig.storeTransposed = co.storeTransposed;
    config.useFloat16LookupTables = co.useFloat16;
    config.usePrecomputedTables = co.usePrecomputed;
    faiss::gpu::GpuIndexIVFPQ index (
    &resources, (int)d, nlist, nsub,8,nedge,256,faiss::METRIC_L2, config);
      std::string prename   = _name+ "_" + std::to_string(d) + "_" + std::to_string(nlist)
                                   + "_" + std::to_string(nsub);
         if(nedge >32){
            prename= prename +  + "_" + std::to_string(nedge);
        }
    { // training
         const std::string codebook_file = prename + ".ppqt";
        std::cout << "codebook exists, reading from " << codebook_file << std::endl;
        index.readCodebookFromFile(prename);
    }

           size_t freee_2;
           size_t total_2;
             cudaSetDevice(dev_no);
            //知道CUDA memory的total，以及free memory 的大小，并且打印出来
            //memory的大小是以bit的形式给出
            cudaMemGetInfo(&freee_2, &total_2 );
            printf("the size of free and total bit11 is %dMB, %dMB\n", freee_2/1024/1024, total_2/1024/1024 );

    {
         const std::string db_file = prename + ".dbIdx";
        std::cout << "db exists, reading from " << db_file << std::endl;
        index.readDbFromFile(prename,300*1000*1000);

        printf ("[%.3f s] done\n", elapsed() - t0);

    }
    cudaMemGetInfo(&freee_2, &total_2 );
    printf("the size of free and total bit12 is %dMB, %dMB\n", freee_2/1024/1024, total_2/1024/1024 );
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

    //size_t kgt; // nb of results per query in the GT
   uint kgt;
    faiss::Index::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors

    {
        printf ("[%.3f s] Loading ground truth for %ld queries\n",
                elapsed() - t0, nq);

        // load ground-truth and convert int to long
      //  size_t nq2;
       //int *gt_int = ivecs_read("/home/data/sift1b/gnd/idx_10M.ivecs", &kgt, &nq2);
       // assert(nq2 == nq || !"incorrect nb of ground truth entries");

          //uint nq;
       // int *gt_int = ivecs_read("/home/data/deep1B/truth.ivecs", &kgt, &nq2);

        	header("sift300M.imem", nq, kgt);
        	std::cout << "nq" << nq << "kgt" << kgt<< std::endl;;
	       int *gt_int = new int[kgt * nq];
//read<int>(path_truth, GT, gt_dim, gt_num);
	    read<int>("sift300M.imem", gt_int, kgt * nq, 0);

        gt = new faiss::Index::idx_t[kgt * nq];
        for(int i = 0; i < kgt * nq; i++) {
            gt[i] = gt_int[i];
        }

          //for (int v = 0; v < kgt ; v++) {
            //std::cout << "\t" << gt[v];
            //}
            //std::cout << "read GT " <<gt[99] << std::endl;
        delete [] gt_int;
    }

    { // searching the database
        int k = 128;
        double start = elapsed();
        printf ("[%.3f s] Searching the %d nearest neighbors "
                "of %ld vectors in the index\n",
                start - t0, k, nq);

        std::vector<faiss::Index::idx_t> nns (k * nq);
        std::vector<float>               dis (k * nq);
        index.setNumProbes(64);

        index.w1_=256;
        std::cout << "index.w1_" <<index.w1_ << std::endl;
       index.search(nq, xq, k, dis.data(), nns.data());

        //index.testDistance_(nq, xq, k, dis.data(), nns.data());
         double end = elapsed();
       printf ("[%.3f s] Query results (vector ids, then distances):\n",
            end - t0);
       printf ("\n");
//           for (int i = 0; i < 10; i++) {
//            printf ("query %2d: ", i);
//            for (int j = 0; j < 10; j++) {
//                printf ("%7ld ", nns[j + i * k]);
//            }
//            printf ("\n     dis: ");
//            for (int j = 0; j < 10; j++) {
//                printf ("%7g ", dis[j + i * k]);
//            }
//            printf ("\n");
//          }
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
