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
    //std::cout << "num: " << num << "  dim: " << dim << std::endl;
	//std::cout << "tellg: " << fin.tellg() << std::endl;
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
     uint d ;
     int nlist=8*1024;
     int nsub = 8;
     int nedge = 32;
     //int tempmem = 1536*1024*1024;
     std::string _name="sift1b";


    // make a set of nt training vectors in the unit cube
    // (could be the database)
    uint nt;
    //float *xt = fvecs_read("/home/data/sift1m/learn.fvecs", &d, &nt);
    header("/home/data/sift1b/learn.umem", nt, d);
    const char *path_learn =
	"/home/data/sift1b/learn.umem";
    float *xt = readUint8(path_learn, d, nlist*32, 0);
    int dev_no = 1;

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
      &resources, (int)d, nlist, nsub,8,nedge,256,faiss::METRIC_L2, config);
          std::string prename   = _name+ "_" + std::to_string(d) + "_" + std::to_string(nlist)
                                   + "_" + std::to_string(nsub);
        if(nedge >32){
            prename= prename +  + "_" + std::to_string(nedge);
        }
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
             index.train (nlist*32,xt);
              index.writeCodebookToFile(prename);
        }else {
            std::cout << "codebook exists, reading from " << codebook_file << std::endl;
            index.readCodebookFromFile(prename);
        }
        //index.readTreeFromFile(nt, xt,"./tmp_128_64_4096_256.ppqt");
        delete [] xt;
    }



    { printf ("[%.3f s] Loading database\n", elapsed() - t0);

       uint nq, nb, d2;
       uint kgt=100;
        float *xq;
        header("/home/data/sift1b/query.umem", nq, d2);
       xq = readUint8("/home/data/sift1b/query.umem", d2, nq,0);
       // xq = fvecs_read("/home/data/sift1m/query.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
        for (int v = 0; v < 128; v++) {
            std::cout << "\t" << xq[v];
            }
            std::cout << "read QuerySet " << xq[99] << std::endl;


        header("/home/data/sift1b/base.umem", nb, d2);
        float *xb;
        //float *xb = fvecs_read("/home/data/sift1m/base.fvecs", &d2, &nb);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf ("[%.3f s] Indexing database, size %ld*%ld\n",
                elapsed() - t0, nb, d);
        printf ("[%.3f s] Adding the vectors to the index\n",
                elapsed() - t0);

        const std::string db_file = prename + ".dbIdx";
        int chuncksize = 100000;
        int batch = 3000;
        int total = batch*chuncksize;

        std::vector<faiss::Index::idx_t> nns (batch*kgt* nq);
        std::vector<float>              dis (batch*kgt* nq);
        int j=0;




        for(int iter = 0;iter<total;iter+=chuncksize){
             xb = readUint8("/home/data/sift1b/base.umem", d2, chuncksize, iter);
            if(iter==0){
                     for (int v = 0; v < 128 ; v++) {
                    std::cout << "\t" << xb[v];
                    }
                    std::cout << "read BASE " <<xb[99] << std::endl;
            }
            std::vector<long> ids(chuncksize);
            for (long i = 0; i < chuncksize; ++i) {
                ids[i] = iter + i;
            }
          index.add_with_ids2(chuncksize,nq,kgt,xb,xq,ids.data(), nns.data()+j*kgt * nq, dis.data()+j*kgt * nq);

                 j++;

                 delete [] xb;

        }
        std::vector<faiss::Index::idx_t> outnns (kgt * nq);
        std::vector<float>               outdis (kgt * nq);
        int batch1=20;
        int tileSize =batch/batch1;

        std::vector<faiss::Index::idx_t> outnns1 (batch1*kgt * nq);
        std::vector<float>               outdis1 (batch1*kgt * nq);

        for(int i =0;i<batch1;i++)
        index.merge(nns.data()+i*tileSize*kgt * nq,dis.data()+i*tileSize*kgt * nq,kgt,nq,tileSize,outdis1.data()+i*kgt * nq,outnns1.data()+i*kgt * nq);

        index.merge(outnns1.data(),outdis1.data(),kgt,nq,batch1,outdis.data(),outnns.data());
        const std::string prename2   = "sift300M.imem" ;
          std::fstream fout(prename2,std::ios_base::out | std::ios_base::binary);

                fout<<nq<<std::endl;
                fout<<kgt<<std::endl;
                fout.ignore();
                fout.seekg(0,std::ios::beg);
                fout.seekg(20,std::ios::beg);


        int* gt_int = new int[kgt * nq];
        for(int i = 0; i < kgt * nq; i++) {
            gt_int[i] = outnns[i];
        }
        fout.write((char*)gt_int,sizeof(int)*kgt* nq);
//         size_t kgt2,nq2;
//        int *gt_int2 = ivecs_read("/home/data/sift1b/gnd/idx_10M.ivecs", &kgt2, &nq2);
//               printf ("\n");
//           for (int i = 0; i < 1; i++) {
//            printf ("gt_int %2d: ", i);
//            for (int j = 0; j < kgt; j++) {
//                printf ("%7ld ", gt_int[j + i * kgt]);
//            }
//            printf ("\n     gt_int2: ");
//            for (int j = 0; j < kgt; j++) {
//                printf ("%7ld ", gt_int2[j + i * kgt2]);
//            }
//            printf ("\n");
//          }
//
//        for(int i = 0; i < nq; i++) {
//           for(int j = 0; j < kgt; j++) {
//                if(gt_int[i*kgt+j]!=gt_int2[i*kgt2+j])
//                 std::cout <<"i: " <<i <<" j: " <<j <<" gt_int: " <<gt_int[i*kgt+j] << " gt_int2: " <<gt_int2[i*kgt2+j] << std::endl;
//            }
//        }




        printf ("[%.3f s] done\n", elapsed() - t0);
    }

    return 0;
}
