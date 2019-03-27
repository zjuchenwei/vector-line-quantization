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
#include <mpi.h>
#include <algorithm>
#include <typeinfo>
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
void merge(float *a,float *b,int len,float *dis,faiss::Index::idx_t *na,faiss::Index::idx_t *nb,faiss::Index::idx_t *nns)
{
    int p=0;//遍历左子序列的游标
    int r=0;//遍历右子序列的游标
    int k=0; //归并结果序列的游标---当前归并元素在结果集中的位置
    while(p<len&&r<len)
    {
        if(a[p]<b[r])
        {
            dis[k]=a[p];
            nns[k]=na[p];
            k++;
            p++;
        }
        else{
            dis[k]=b[r];
            nns[k]=nb[r];
            k++;
            r++;
        }
    }

    if(p>len-1)
    {
        for(int i=r;i<len;i++)
        {
            dis[k]=b[i];
            nns[k]=nb[i];
            k++;
        }
    }
    else{
        for(int i=p;i<len;i++)
        {
            dis[k]=a[i];
            nns[k]=na[i];
            k++;
        }
    }
}
int main (int argc, char* argv[])
{

    int rank, numproces;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);//获得进程号
	MPI_Comm_size(MPI_COMM_WORLD, &numproces);//返回通信的进程数


    double t0 = elapsed();

    // dimension of the vectors to index
    uint d ;
     int nlist=64*1024;
     int nsub = 16;
     int nedge = 64;
     int tempmem = 1800*1024*1024;
     std::string _name="deep1b";

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    uint nt;
    header("/home/data/deep1B/learn.umem", nt, d);
   // float *xt = fvecs_read("/home/data/sift1m/learn.fvecs", &d, &nt);
    int dev_no = rank;
    /*
    printf ("[%.3f s] Begin d=%d nb=%ld nt=%nt dev_no=%d\n",
            elapsed() - t0, d, nb, nt, dev_no);
    */
    // a reasonable number of centroids to index nb vectors

    faiss::gpu::StandardGpuResources resources;
    //resources.setTempMemory(tempmem);
    resources.setTempMemoryFraction(0.25);

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
      // if(rank==1){
         //  prename="/home/data/rank1/"+prename;

       // }
       // else{
       //      prename="/home/data/deep1B/"+prename;

       // }
    { // training
         const std::string codebook_file = prename + ".ppqt";
        std::cout << "codebook exists, reading from " << codebook_file << std::endl;
        index.readCodebookFromFile(prename);
    }


    {
         const std::string db_file = prename + ".dbIdx";
        std::cout << "db exists, reading from " << db_file << std::endl;
        index.readDbFromFile(prename,0,numproces,rank);
        printf ("[%.3f s] done\n", elapsed() - t0);
    }

    uint nq;
    float *xq;

    {
        printf ("[%.3f s] Loading queries\n", elapsed() - t0);

        uint d2;

       header("/home/data/deep1B/query.umem", nq, d2);
       xq = readFloat("/home/data/deep1B/query.umem", d2, nq,0);
       // xq = fvecs_read("/home/data/sift1m/query.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
        for (int v = 0; v < 128; v++) {
            std::cout << "\t" << xq[v];
            }
            std::cout << "read QuerySet " << xq[99] << std::endl;

    }

    uint kgt; // nb of results per query in the GT
    faiss::Index::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors

    {
        printf ("[%.3f s] Loading ground truth for %ld queries\n",
                elapsed() - t0, nq);

        // load ground-truth and convert int to long
        uint nq2;
        //int *gt_int = ivecs_read("/home/data/deep1B/truth.ivecs", &kgt, &nq2);

        	header("/home/data/deep1B/truth.imem", nq2, kgt);
	       int *gt_int = new int[kgt * nq2];
//	read<int>(path_truth, GT, gt_dim, gt_num);
	     read<int>("/home/data/deep1B/truth.imem", gt_int, kgt * nq2, 0);



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
         std::vector<faiss::Index::idx_t> nns (k * nq);
         std::vector<float>              dis (k * nq);


        if(rank==0)
          {
             nns.resize(k * nq*numproces);
             dis.resize(k * nq*numproces);
          }
        MPI_Barrier(MPI_COMM_WORLD);
        index.setNumProbes(nlist/1024);
        //index.w1_=nlist*nedge/2/1024;
        index.setNumProbes(nlist/1024);
         index.w1_=nlist*nedge*0.25/1024;
         std::cout << " index.w1_ " <<  index.w1_ << std::endl;
        double start = elapsed();
       printf ("[%.3f s] Searching the %d nearest neighbors "
                "of %ld vectors in the index\n",
                start - t0, k, nq);
       index.search(nq, xq, k, dis.data(), nns.data());

        //index.testDistance_(nq, xq, k, dis.data(), nns.data());
         double end = elapsed();
       printf ("[%.3f s] Query results (vector ids, then distances):\n",
            end - t0);
       printf ("\n");

          if(rank!=0)
          {
              printf("++++++++++++++++++++++++++++++sending\n");
              MPI_Send(nns.data(), k*nq*sizeof(faiss::Index::idx_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
              MPI_Send(dis.data(), k*nq, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
          }

          //std::cout << "----------------nns *"<<typeid(nns.data()).name() << std::endl;

          if(rank==0)
          {

                  MPI_Status status;
                  std::vector<faiss::Index::idx_t> outnns (k * nq);
                  std::vector<float>               outdis (k * nq);

                  printf("**************************recving\n");
                  for(int i=1;i<numproces;i++){
                     MPI_Recv(nns.data()+i*k * nq,nq*k*sizeof(faiss::Index::idx_t),MPI_BYTE,i,0,MPI_COMM_WORLD,&status );
                     MPI_Recv(dis.data()+i*k * nq,nq*k,MPI_FLOAT,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

                  }


                 index.merge(nns.data(),dis.data(),k,nq,numproces,outdis.data(),outnns.data());
                 double endtime = elapsed();



                  faiss::Index::idx_t *I = outnns.data();
                  //int *I = tmp.data();
                  int n_1 = 0, n_10 = 0, n_100 = 0,n_200=0;
                    for(int i = 0; i < nq; i++) {
                        int gt_nn = gt[i * kgt];
                        for(int j = 0; j <k; j++) {
                            if (I[i * k + j] == gt_nn) {
                                if(j < 1) n_1++;
                                if(j < 10) n_10++;
                                if(j < 100) n_100++;
                                if(j < 200) n_200++;
                            }
                        }
                    }
                    printf("R@1 = %.4f\n", n_1 / float(nq));
                    printf("R@10 = %.4f\n", n_10 / float(nq));
                    printf("R@100 = %.4f\n", n_100 / float(nq));
                    printf("R@200 = %.4f\n", n_200 / float(nq));


                    std::cout <<numproces <<" GPU avg. query time   " << (endtime-start)*1000 / static_cast<double>(nq) << "ms" << std::endl;

          }

    // faiss::Index::idx_t *I = nns.data();
        // int n_1 = 0, n_10 = 0, n_100 = 0;
        // for(int i = 0; i < nq; i++) {
            // int gt_nn = gt[i * kgt];
            // for(int j = 0; j <k; j++) {
                // if (I[i * k + j] == gt_nn) {
                    // if(j < 1) n_1++;
                    // if(j < 10) n_10++;
                    // if(j < 100) n_100++;
                // }
            // }
        // }
        // printf("R@1 = %.4f\n", n_1 / float(nq));
        // printf("R@10 = %.4f\n", n_10 / float(nq));
        // printf("R@100 = %.4f\n", n_100 / float(nq));
        // std::cout << "avg. query time   " << (end-start)*1000 / static_cast<double>(nq) << "ms" << std::endl;
        // std::cout << "total. query time " << (end-start) << "s" << std::endl;
    }
    delete [] xq;
    delete [] gt;
    MPI_Finalize();
    return 0;
}
