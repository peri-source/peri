#ifndef __UTIL_H__
#define __UTIL_H__

#include <curand.h>
#include "types.h"

#define S2PI 0.79788456080286541f
#define ME 1.1920929e-7f

typedef unsigned long long int ullong;
extern ullong vseed;
extern ullong vran;
extern curandGenerator_t gen;

float ran_ran2();
void ran_seed(long j);

void initializeDevice(int device);
void destroyDevice();

void setSeed(int seed);
float *createDeviceArray(int len);
double *createDeviceArrayD(int len);
cfloat *createDeviceArrayC(int len);
int *createDeviceArrayInt(int len);
void freeDeviceArray(float *f);
void freeDeviceArrayD(double *f);
void freeDeviceArrayC(cfloat *f);
void freeDeviceArrayInt(int *f);

void copyToDevice(int l, float *i, float *dev);
void copyFromDevice(float *dev, int l, float *o);

void copyToDeviceD(int l, double *i, double *dev);
void copyFromDeviceD(double *dev, int l, double *o);

void copyToDeviceInt(int total_len, int *inpy, int *dev);
void copyFromDeviceInt(int *dev, int total_len, int *onpy);

void copyToDeviceC(int l, cfloat *i, cfloat *dev);
void copyFromDeviceC(cfloat *dev, int l, cfloat *o);

void copyArray(float *src, int l, float *dst);
void copyArrayInt(int *src, int l, int *dst);

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

#endif
