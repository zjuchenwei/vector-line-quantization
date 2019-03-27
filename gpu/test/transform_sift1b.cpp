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

#include "../../VectorTransform.h"
#include <iostream>
#include<fstream>
#include <stdexcept>
#include "../GpuAutoTune.h"
#include "../../index_io.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
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
float* readUint8(const char* _fn, size_t _dim, size_t _num, size_t _offset) {

    size_t offset = _offset * _dim;
    size_t length = _num * _dim;

    float * buf = new float[length];

    uint8_t *raw_data = new uint8_t[length];
    read<uint8_t>(_fn, raw_data, length, offset);
    for (int i = 0; i < length; i++)
        buf[i] = (float)raw_data[i];

    delete[] raw_data;

    return buf;
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
bool file_exists(const std::string& _name) {
	struct stat buffer;

	return (stat(_name.c_str(), &buffer) == 0);
}

void transform_matrix (const char *fname,const char *fname2,
                    size_t *d_out, size_t *n_out)
{

    uint d;
    uint n;
    int d2 = 32;
    int zero = 0;

    header(fname,n,d);

   cudaSetDevice(0);
   cudaSetDeviceFlags (cudaDeviceMapHost);

    std::fstream fout(fname2,std::ios_base::out | std::ios_base::binary);

    fout<<n<<std::endl;
    fout<<d2<<std::endl;
    fout.ignore();
    fout.seekg(0,std::ios::beg);
    fout.seekg(20,std::ios::beg);

    printf("n:%d d:%d\n",n,d);
    *d_out = d;
    *n_out = n;
    printf("read starting...\n");
    int chuncksize = 100000;
    int num = 0;
    float *xb;
    const std::string prename="sift1m";
    const std::string fmatrix= prename+"_"+std::to_string(d) + "_" +std::to_string(d2)+".matrix";
    faiss::OPQMatrix desc(d, 8, d2);

    if (!file_exists(fmatrix)) {
        xb = readFloat(fname, d, chuncksize, 0);

        desc.train(chuncksize,xb);
        std::ofstream fppqt(fmatrix.c_str(), std::ofstream::out | std::ofstream::binary);
        fppqt.write((char*) desc.A.data(),d*d2* sizeof(float));
        fppqt.close();
         delete [] xb;
         return;
    }else {
          std::ifstream f(fmatrix.c_str(), std::ofstream::in | std::ofstream::binary);
          desc.A.resize(d*d2);
          f.read((char*) desc.A.data(),d*d2* sizeof(float));
          desc.is_trained = true;
          desc.is_orthonormal = true;
          f.close();
    }

     float alpha=1.0;
     float beta=0.0;
     cublasHandle_t handle;
     cublasCreate(&handle);

     float *d_R;
     checkCudaErrors(cudaMalloc(&d_R,d*d2*sizeof(float)));
     checkCudaErrors(cudaMemcpy(d_R,desc.A.data(),d*d2*sizeof(float),cudaMemcpyHostToDevice));
     float *d_xb;
     float *d_xtf;
    chuncksize = 100000;
    chuncksize = std::min(chuncksize, (int)n);
    checkCudaErrors(cudaMalloc(&d_xb,chuncksize*d*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_xtf,chuncksize*d2*sizeof(float)));
    float* xt_f=new float[chuncksize * d2];
        for(int iter = 0;iter<n;iter+=chuncksize){
            int len = std::min(chuncksize, (int)(n - iter));
            xb = readFloat(fname, d, len, iter);
            for(int i=0;i<1000;i++)
                printf("%.4f ", xb[i]);
            printf("%d\n", iter);
            checkCudaErrors(cudaMemcpy(d_xb,xb,chuncksize*d*sizeof(float),cudaMemcpyHostToDevice));
            //desc.reverse_transform(len,xb,xt_f);
            cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,chuncksize,d2,d,&alpha,d_xb,d,d_R,d2,&beta,d_xtf,chuncksize);
           cudaMemcpy(xt_f,  d_xtf, chuncksize*d2*sizeof(float), cudaMemcpyDeviceToHost);
            printf("\n");
             fout.write((char*)xt_f,sizeof(float)*len*d2);
             delete [] xb;
        }
    fout.close();
    delete [] xt_f;
    printf("read end\n");
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
    //return (int*)fvecs_read(fname, d_out, n_out);
}


double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}


int main ()
{

    double t0 = elapsed();

    // dimension of the vectors to index
    size_t d ;


    // make a set of nt training vectors in the unit cube
    // (could be the database)
    size_t nt;
    transform_matrix("/home/data/sift1m/base.umem", "/home/data/sift1m/base32.umem", &d, &nt);
    printf("nt:%d  d:%d\n",nt,d);
    //nt = 1000000;
    //xt = fvecs_read("/home/data/sift1b/base.umem", "/home/data/sift1b/base32.umem", &d, &nt);
    //delete [] xt;
    return 0;

}
