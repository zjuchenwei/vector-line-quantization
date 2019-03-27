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
     int nsub = 64;
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
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = dev_no;
    faiss::gpu::GpuClonerOptions co;
    co.useFloat16 = true;
    config.indicesOptions = co.indicesOptions;
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
           // index.readTreeFromFile(nt, xt,"./tmp_128_64_4096_256.ppqt");
        }
        //index.readTreeFromFile(nt, xt,"./tmp_128_64_4096_256.ppqt");
        delete [] xt;
    }

    { // I/O demo
        const char *outfilename = "./index_trained.faissindex";
        printf ("[%.3f s] storing the pre-trained index to %s\n",
                elapsed() - t0, outfilename);

        faiss::Index * cpu_index = faiss::gpu::index_gpu_to_cpu (&index);

        write_index (cpu_index, outfilename);

        delete cpu_index;
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
        if (!file_exists(db_file)) {
          std::cout << "db not exists " << db_file << std::endl;
            index.add (nb, xb);
            index.writeDbToFile(prename);

        }else {
            std::cout << "db exists, reading from " << db_file << std::endl;
            index.readDbFromFile(prename);
        }
        printf ("[%.3f s] done\n", elapsed() - t0);
    }
    size_t nq;
    float *xq;

    {
        printf ("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read("/home/data/sift1m/query.fvecs", &d2, &nq);
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
        int *gt_int = ivecs_read("/home/data/sift1m/groundtruth.ivecs", &kgt, &nq2);
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

    { // searching the database
        int k = 128;
        double start = elapsed();
        printf ("[%.3f s] Searching the %d nearest neighbors "
                "of %ld vectors in the index\n",
                start - t0, k, nq);

        std::vector<faiss::Index::idx_t> nns (k * nq);
        std::vector<float>               dis (k * nq);
        index.setNumProbes(64);
       index.search(nq, xq, k, dis.data(), nns.data());

        //index.testDistance_(nq, xq, k, dis.data(), nns.data());
         double end = elapsed();
       printf ("[%.3f s] Query results (vector ids, then distances):\n",
            end - t0);
       printf ("\n");
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
