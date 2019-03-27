#ifndef FILEHELPER_HPP
#define FILEHELPER_HPP

#include <string>
#include <iostream>
#include <fstream>

namespace faiss {
/* read Header from fvecs */
template<typename T>
T* readJegou(const char *path, uint &n, uint &d) ;

template<typename T>
void readJegouHeader(const char *path, uint &n, uint &d);

template<>
uint8_t* readJegou<uint8_t>(const char *path, uint &n, uint &d);


uint8_t* readBatchJegou(const char *path, uint start_pos, uint num);

template<>
void readJegouHeader<uint8_t>(const char *path, uint &n, uint &d);


template<typename T = uint8_t>
void write(std::string fs, size_t num, uint dim, T *ptr, size_t len, size_t offset = 0) ;




template<typename T = uint8_t>
void read(std::string fs, size_t &num, uint &dim, T *ptr, size_t len, size_t offset = 0);

void header(std::string fs, uint &num, uint &dim) ;

void writeFloat(std::string _fn, size_t _dim, size_t _num, float* _x, size_t _offset);
void writeInt(std::string _fn, size_t _dim, size_t _num, int* _x, size_t _offset);

float* readFloat(const char* _fn, size_t _dim, size_t _num, size_t _offset) ;

float* readUint8(const char* _fn, size_t _dim, size_t _num, size_t _offset) ;

int* readInt(const char* _fn, size_t _dim, size_t _num, size_t _offset);
#endif
}
