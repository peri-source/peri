#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <unistd.h>

#include <cuda.h>
#include <curand.h>

#include "util.h"
#include "error.h"

ullong vseed;
ullong vran;
curandGenerator_t gen;
int *_derrno;
int _herrno;

//=================================================
// random number generator
//=================================================
void ran_seed(long j){
  vseed = j;  vran = 4101842887655102017LL;
  vran ^= vseed;
  vran ^= vran >> 21; vran ^= vran << 35; vran ^= vran >> 4;
  vran = vran * 2685821657736338717LL;
}

float ran_ran2(){
    vran ^= vran >> 21; vran ^= vran << 35; vran ^= vran >> 4;
    ullong t = vran * 2685821657736338717LL; return 5.42101086242752217e-20*t;
}

void initializeDevice(int device){
    cudaSetDevice(device);
    CSAFECALL( cudaMalloc((void**)&_derrno, sizeof(int)) );
    CSAFECALL( cudaMemset(_derrno, 0, sizeof(int)) );
}

void destroyDevice(){
    curandDestroyGenerator(gen);
    cudaDeviceReset();
}

void setSeed(int seed){
    ran_seed(seed);
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    //curandDestroyGenerator(gen);
}

float *createDeviceArray(int len){
    float *out;
    CSAFECALL( cudaMalloc((void**) &out, sizeof(float)*len) );
    cudaMemset(out, 0, sizeof(float)*len);
    return out;
}
double *createDeviceArrayD(int len){
    double *out;
    CSAFECALL( cudaMalloc((void**) &out, sizeof(double)*len) );
    cudaMemset(out, 0, sizeof(double)*len);
    return out;
}
cfloat *createDeviceArrayC(int len){
    cfloat *out;
    CSAFECALL( cudaMalloc((void**) &out, sizeof(cfloat)*len) );
    cudaMemset(out, 0, sizeof(cfloat)*len);
    return out;
}
int *createDeviceArrayInt(int len){
    int *out;
    CSAFECALL( cudaMalloc((void**) &out, sizeof(int)*len) );
    cudaMemset(out, 0, sizeof(int)*len);
    return out;
}

void freeDeviceArray(float *f){ CSAFECALL(cudaFree(f)); }
void freeDeviceArrayD(double *f){ CSAFECALL(cudaFree(f)); }
void freeDeviceArrayC(cfloat *f){ CSAFECALL(cudaFree(f)); }
void freeDeviceArrayInt(int *f){ CSAFECALL(cudaFree(f)); }

void copyToDevice(int total_len, float *inpy, float *dev){
    CSAFECALL( cudaMemcpy(dev, inpy, sizeof(float)*total_len, cudaMemcpyHostToDevice) );
}

void copyFromDevice(float *dev, int total_len, float *onpy){
    CSAFECALL( cudaMemcpy(onpy, dev, sizeof(float)*total_len, cudaMemcpyDeviceToHost) );
}

void copyToDeviceD(int total_len, double *inpy, double *dev){
    CSAFECALL( cudaMemcpy(dev, inpy, sizeof(double)*total_len, cudaMemcpyHostToDevice) );
}

void copyFromDeviceD(double *dev, int total_len, double *onpy){
    CSAFECALL( cudaMemcpy(onpy, dev, sizeof(double)*total_len, cudaMemcpyDeviceToHost) );
}

void copyToDeviceInt(int total_len, int *inpy, int *dev){
    CSAFECALL( cudaMemcpy(dev, inpy, sizeof(int)*total_len, cudaMemcpyHostToDevice) );
}

void copyFromDeviceInt(int *dev, int total_len, int *onpy){
    CSAFECALL( cudaMemcpy(onpy, dev, sizeof(int)*total_len, cudaMemcpyDeviceToHost) );
}

void copyToDeviceC(int total_len, cfloat *inpy, cfloat *dev){
    CSAFECALL( cudaMemcpy(dev, inpy, sizeof(cfloat)*total_len, cudaMemcpyHostToDevice) );
}

void copyFromDeviceC(cfloat *dev, int total_len, cfloat *onpy){
    CSAFECALL( cudaMemcpy(onpy, dev, sizeof(cfloat)*total_len, cudaMemcpyDeviceToHost) );
}


void copyArray(float *src, int len, float *dst){
    memcpy(dst, src, sizeof(float)*len);
}

void copyArrayInt(int *src, int len, int *dst){
    memcpy(dst, src, sizeof(int)*len);
}
