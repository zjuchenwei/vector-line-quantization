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
    size_t d ;
     int nlist=1024;
     int nsub = 8;
     std::string _name="tmp";

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    size_t nt;
    float *xt = fvecs_read("/home/data/sift1m/learn.fvecs", &d, &nt);
    int dev_no = 0;
    /*
    printf ("[%.3f s] Begin d=%d nb=%ld nt=%nt dev_no=%d\n",
            elapsed() - t0, d, nb, nt, dev_no);
    */
    // a reasonable number of centroids to index nb vectors

    faiss::gpu::StandardGpuResources resources;


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
    config.indicesOptions =  faiss::gpu::INDICES_CPU;
    config.flatConfig.useFloat16 = co.useFloat16CoarseQuantizer;
    config.flatConfig.storeTransposed = co.storeTransposed;
    config.useFloat16LookupTables = co.useFloat16;
    config.usePrecomputedTables = co.usePrecomputed;
    faiss::gpu::GpuIndexIVFPQ index (
      &resources, (int)d, nlist, nsub,8,32,1,faiss::METRIC_L2, config);
     const std::string prename   = _name+ "_" + std::to_string(d) + "_" + std::to_string(nlist)
                                   + "_" + std::to_string(nsub);
    { // training
        printf ("[%.3f s] Generating %ld vectors in %dD for training\n",
                elapsed() - t0, nt, d);

        printf ("[%.3f s] Training the index\n",
                elapsed() - t0);
        index.verbose = true;
         for (int v = 0; v < 128; v++) {
            std::cout << "\t" << xt[v];
            }
            std::cout << "read TrainSet " << xt[123] << std::endl;
            const std::string codebook_file = prename + ".ppqt";
        if (!file_exists(codebook_file)) {
            std::cout << "codebook not exists, writing to " << codebook_file << std::endl;
             index.train (nt, xt);
              index.writeCodebookToFile(prename);
        }else {
            std::cout << "codebook exists, reading from " << codebook_file << std::endl;
            index.readCodebookFromFile(prename);
        }
        //index.readTreeFromFile(nt, xt,"./tmp_128_64_4096_256.ppqt");
        delete [] xt;
    }



    { printf ("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        float *xb = fvecs_read("/home/data/sift1m/base.fvecs", &d2, &nb);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf ("[%.3f s] Indexing database, size %ld*%ld\n",
                elapsed() - t0, nb, d);
        printf ("[%.3f s] Adding the vectors to the index\n",
                elapsed() - t0);
        for (int v = 0; v < 128 ; v++) {
            std::cout << "\t" << xb[v];
            }
            std::cout << "read BASE " <<xb[99] << std::endl;

        const std::string db_file = prename + ".dbIdx";
        index.add (nb, xb);
        index.writeDbToFile(prename);


        printf ("[%.3f s] done\n", elapsed() - t0);
    }

    return 0;
}
