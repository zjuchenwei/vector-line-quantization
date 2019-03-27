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





#include "../StandardGpuResources.h"
#include "../GpuIndexIVFPQ.h"
#include "../../filehelper.h"
#include "../GpuAutoTune.h"
#include "../../index_io.h"
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

int main ()
{

    double t0 = elapsed();

    // dimension of the vectors to index
     uint d ;
     int nlist=32*1024;
     int nsub = 8;
     //int tempmem = 1536*1024*1024;
     std::string _name="sift1b";
    // make a set of nt training vectors in the unit cube
    // (could be the database)
    uint nt;
    //float *xt = fvecs_read("/home/data/sift1m/learn.fvecs", &d, &nt);
    faiss::header("/home/data/sift1b/learn.umem", nt, d);
    const char *path_learn =
	"/home/data/sift1b/learn.umem";
    float *xt = faiss::readUint8(path_learn, d, 2000000, 0);
    int dev_no = 0;


    //printf ("[%.3f s] Begin d=%d nt=%nt dev_no=%d\n",
       //     elapsed() - t0, d, nt, dev_no);
    /*
    printf ("[%.3f s] Begin d=%d nb=%ld nt=%nt dev_no=%d\n",
            elapsed() - t0, d, nb, nt, dev_no);
    */
    // a reasonable number of centroids to index nb vectors

    faiss::gpu::StandardGpuResources resources;
   // resources.setTempMemory(tempmem);

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
    co.useFloat16 = true;
    config.indicesOptions = faiss::gpu::INDICES_CPU;
    config.flatConfig.useFloat16 = co.useFloat16CoarseQuantizer;
    config.flatConfig.storeTransposed = co.storeTransposed;
    config.useFloat16LookupTables = co.useFloat16;
    config.usePrecomputedTables = co.usePrecomputed;
    faiss::gpu::GpuIndexIVFPQ index (
      &resources, (int)d, nlist, nsub,8,32,256,faiss::METRIC_L2, config);
     const std::string prename   = _name+ "_" + std::to_string(d) + "_" + std::to_string(nlist)
                                   + "_" + std::to_string(nsub);
    { // training
        printf ("[%.3f s] Generating %ld vectors in %dD for training\n",
                elapsed() - t0, nt, d);

        printf ("[%.3f s] Training the index\n",
                elapsed() - t0);
        index.verbose = true;
         for (int v = 0; v <128; v++) {
            std::cout << "\t" << xt[v];
            }
            std::cout << "read TrainSet " << xt[123] << std::endl;
            const std::string codebook_file = prename + ".ppqt";
        if (!file_exists(codebook_file)) {
            std::cout << "codebook not exists, writing to " << codebook_file << std::endl;
             index.train (2000000, xt);
              index.writeCodebookToFile(prename);
        }else {
            std::cout << "codebook exists, reading from " << codebook_file << std::endl;
            index.readCodebookFromFile(prename);
        }

         const std::string prename1   = "centroids_sift1b_" + std::to_string(d) + "_" + std::to_string(nlist)
                                   + "_" + std::to_string(nsub);
          index.writeCentroidsToFile(prename1);
        //index.readTreeFromFile(nt, xt,"./tmp_128_64_4096_256.ppqt");
    }



    { printf ("[%.3f s] Loading database\n", elapsed() - t0);

        uint nb, d2;

        faiss::header("/home/data/sift1b/base.umem", nb, d2);
        float *xb;
        //float *xb = fvecs_read("/home/data/sift1m/base.fvecs", &d2, &nb);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf ("[%.3f s] Indexing database, size %ld*%ld\n",
                elapsed() - t0, nb, d);
        printf ("[%.3f s] Adding the vectors to the index\n",
                elapsed() - t0);

        const std::string db_file = prename + ".dbIdx";
        int chuncksize =1000000;
        const std::string prename2   = "precomputed_idxs_sift1b_" + std::to_string(d) + "_" + std::to_string(nlist)
                                   + "_" + std::to_string(nsub)+ ".imem" ;
          std::fstream fout(prename2,std::ios_base::out | std::ios_base::binary);

                fout<<nb<<std::endl;
                fout<<1<<std::endl;
                fout.ignore();
                fout.seekg(0,std::ios::beg);
                fout.seekg(20,std::ios::beg);

        for(int iter = 0;iter<100000000;iter+=chuncksize){
             xb = faiss::readUint8("/home/data/sift1b/base.umem", d2, chuncksize, iter);
            if(iter==0){
                     for (int v = 0; v < 128 ; v++) {
                    std::cout << "\t" << xb[v];
                    }
                    std::cout << "read BASE " <<xb[99] << std::endl;
            }
            std::vector<long> ids(chuncksize);
              for (long i = 0; i < chuncksize; ++i) {
                ids[i] = index.ntotal + i;
              }
               int* assgin1 = index.add_with_ids1(chuncksize, xb, ids.data());
                fout.write((char*)assgin1,sizeof(int)*chuncksize);


                 delete[] assgin1;
                 delete [] xb;

        }
            fout.close();

             for(int iter = 0;iter<1000000;iter+=512*1024){
         int* xb1 = faiss::readInt("precomputed_idxs_sift1b.imem", 1, 15, iter);

                 for (int v = 0; v < 15 ; v++) {
                std::cout << "\t" << xb1[v];
                }
                std::cout << "read BASE " <<xb1[14] << std::endl;
         delete [] xb1;
             }

        printf ("[%.3f s] done\n", elapsed() - t0);
    }

    return 0;
}
