#ifndef __NBL_H__
#define __NBL_H__

#include "types.h"
//#include "error.h"

#define ACTIVE   0
#define INACTIVE 1
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

void naive_renormalize_radii(int NX, double *x, int NR, double *r, double zscale, int doprint);
double naive_overlap(int NX, double *x, int NR, double *r, double zscale, int doprint);

class NBL {
public:
    NBL(int N);
    ~NBL();

    void reset();
    void build_rij();
    void build_cells();

private:
    int N;
    double *pos;
    double *rad;
    double *rij;
    int *type;

    int *neighbors;
    int *nneighs;
    int *noverlaps;

    int *cells;
    int *count;

    int *size;
    double *BL;
    double *BU;
    int ncells;
    int max_particles_per_cell;
    double cutoff;
    double cutoffsq;
    int doerror;

};

#endif
