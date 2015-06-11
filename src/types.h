#ifndef __TYPES_H__
#define __TYPES_H__

#ifdef CUDA
#include <cuComplex.h>
#define cfloat cuFloatComplex
#endif

typedef unsigned int uint;
typedef unsigned long long int ullong;

#ifdef CUDA
typedef int3 dimvec;
#endif

typedef struct {
    int plot_sizex;
    int plot_sizey;
    int win;
    int dopoints;
    int keys[256];

    float L[3];
    float pradius;
    float level;
    float focus;
} plotconf;

typedef struct {
    int N;
    int inuse;
    int dim;
    int stage;
    float t, dt;
    float T;
    float damp;
    float force_hard_sphere;
    float force_hertz_epsilon;

    float *x;
    float *v;
    float *f;
    float *rad;
    int *type;
} simsys;

typedef struct {
    float *rij;
    float *forces;
    int *neighbors;
    int *nneighs;
    int *overlaps;

    int *cells;
    int *count;

    int *size;
    int *pbc;
    float *BL;
    float *BU;
    int N;
    int ncells;
    int max_particles_per_cell;
    float cutoff;
    float cutoffsq;
    int doerror;
} nbl;

#define FIELDTYPE(ARRTYPE, NAME) \
    typedef struct {        \
        ARRTYPE *arr;       \
        int dim;            \
        int len;            \
        dimvec size;        \
    } NAME;

#ifdef CUDA
FIELDTYPE(int, ifield);
FIELDTYPE(float, field);
FIELDTYPE(cfloat, cfield);
#endif

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

#endif
