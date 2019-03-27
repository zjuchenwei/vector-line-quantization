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

int main ()
{

    double t0 = elapsed();

    // dimension of the vectors to index
    uint d ;
     int nlist=64*1024;
     int nsub = 8;
     int nedge = 64;
     int tempmem = 1536*1024*1024;
     std::string _name="deep1b";

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    uint nt;
    header("/home/data/deep1B/learn.umem", nt, d);
   // float *xt = fvecs_read("/home/data/sift1m/learn.fvecs", &d, &nt);
    int dev_no = 0;
    /*
    printf ("[%.3f s] Begin d=%d nb=%ld nt=%nt dev_no=%d\n",
            elapsed() - t0, d, nb, nt, dev_no);
    */
    // a reasonable number of centroids to index nb vectors

 std::string prename   = _name+ "_" + std::to_string(d) + "_" + std::to_string(nlist)
                                   + "_" + std::to_string(nsub);
    { // training
          if(nedge >32){
            prename= prename +  + "_" + std::to_string(nedge);
        }


    }
 std::ofstream outFile;
	outFile.open("vlq-adc-deep1b.csv", std::ios::out); // 打开模式可省略

    const std::string dbName   = prename+".dbIdx";
   std::ifstream fdbIdx(dbName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string codeName   =prename+".dbcodes";
   std::ifstream fdbcodes(codeName.c_str(), std::ofstream::in | std::ofstream::binary);

    const std::string countName   = prename+".dbcount";
   std::ifstream fdbcount(countName.c_str(), std::ofstream::in | std::ofstream::binary);

   const std::string laName   = prename+".dblas";
   std::ifstream fdblas(laName.c_str(), std::ofstream::in | std::ofstream::binary);

     std::vector < std::vector<long> > ids;
    std::vector <int> counts;
     ids.resize(nlist*nedge);
     counts.resize(nlist*nedge);
     fdbcount.read((char*)counts.data(), counts.size()* sizeof(int));
     std::vector<uint> binHist;
	binHist.resize(8);
	uint total = 0;

	uint maxVal = 0;
	uint maxIdx = 0;
    for (int i = 0; i < nlist*nedge; ++i) {
       int length = counts[i];

       outFile<< length<<",";


      if((i+1)%10==0)
        outFile<<std::endl;

       if (counts[i] > maxVal) {
			maxVal = counts[i];
			maxIdx = i;
		}

       if (counts[i] == 0)
			binHist[0]++;
		else if (counts[i] < 100)
			binHist[1]++;
        else if (counts[i] < 300)
			binHist[2]++;
		else if (counts[i] < 500)
			binHist[3]++;
		else
			binHist[4]++;

		total += counts[i];

    }
     outFile <<  std::endl;
     outFile.close();
    std::cout << "total entries: " << total << std::endl;

	 std::cout << "histogram: " << std::endl;
	 std::cout << "0 \t" << binHist[0] << std::endl;
	 std::cout << "<100 \t" << binHist[1] << std::endl;
    std::cout << "<300 \t" << binHist[2] << std::endl;
	 std::cout << "<500 \t" << binHist[3] << std::endl;
	 std::cout << ">500 \t" << binHist[4] << std::endl;

	 std::cout << "maxbin: " << maxIdx << "  entries: " << maxVal << std::endl;

     for(int a =0 ; a<8;a++){
                 std::cout << binHist[a]/static_cast<double>(nlist*nedge) << ",";
        }

    std::cout << std::endl;

    fdbIdx.close();
    fdbcodes.close();
    fdbcount.close();
    fdblas.close();
    return 0;
}
