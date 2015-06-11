#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <unistd.h>

#include "error.h"
#include "nbl.h"

double dot(double *a, double *b){
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

double naive_overlap(int NX, double *x, int NR, double *r, int doprint){
    if (NX != 3*NR)
        log_err("Wrong number of parameters\n");

    int N = NR;
    int overlaps = 0;
    for (int i=0; i<N-1; i++){
        for (int j=i+1; j<N; j++){
            double *x0 = &x[3*i];
            double *x1 = &x[3*j];
            double dr[3] = {0,0,0};
            dr[0] = x1[0] - x0[0];
            dr[1] = x1[1] - x0[1];
            dr[2] = x1[2] - x0[2];

            double d = sqrt(dot(dr, dr));

            if (d < r[i] + r[j]){
                if (doprint)
                    printf("%i %i: %f %f %f : %e\n", i, j, r[i], r[j], d, d - (r[i]+r[j]));
                overlaps++;
            }
        }
    }

    return overlaps;
}

void naive_renormalize_radii(int NX, double *x, int NR, double *r, int doprint){
    if (NX != 3*NR)
        log_err("Wrong number of parameters\n");

    int N = NR;
    for (int i=0; i<N-1; i++){
        for (int j=i+1; j<N; j++){
            double *x0 = &x[3*i];
            double *x1 = &x[3*j];
            double dr[3] = {0,0,0};
            dr[0] = x1[0] - x0[0];
            dr[1] = x1[1] - x0[1];
            dr[2] = x1[2] - x0[2];

            double d = sqrt(dot(dr, dr));
            double diff = (d - (r[i] + r[j]));
            if (diff < 0){
                if (doprint)
                    printf("Fixing %i %i : %f %f %e\n", i, j, r[i], r[j], d);
                r[i] -= fabs(diff)/2 + 1e-10;//fabs(diff) / 2 + 1e-10;
                r[j] -= fabs(diff)/2 + 1e-10;//fabs(diff) / 2 + 1e-10;
            }
        }
    }
}

NBL::NBL(int n){
    N = n;
}

NBL::~NBL(){

}

void NBL::reset(){
    memset(cells, 0, sizeof(double)*ncells*max_particles_per_cell);
    memset(count, 0, sizeof(double)*ncells);
    memset(nneighs, 0, sizeof(double)*N);
    memset(noverlaps, 0, sizeof(double)*N);
}

void NBL::build_cells(){
    for (int i=0; i<N; i++){
        int index[3];
        index[0] = (int)((pos[3*i+0]-BL[0])/(BU[0]-BL[0]) * size[0]);
        index[1] = (int)((pos[3*i+1]-BL[1])/(BU[1]-BL[1]) * size[1]);
        index[2] = (int)((pos[3*i+2]-BL[2])/(BU[2]-BL[2]) * size[2]);
        if (index[0] < 0 || index[1] < 0 || index[2] < 0 || index[0] >= size[0]
                || index[1] >= size[1] || index[2] >= size[2])
            return;
        int t  = index[0] + index[1]*size[0] + index[2]*size[0]*size[1];
        unsigned int ct = count[t];
        unsigned int bt = max_particles_per_cell*t + ct;
        cells[bt] = i;
        count[t]++;
    }
}

void NBL::build_rij(){
    for (int i=0; i<N; i++){
        int tt[3];
        int index[3];
        double dx[3];

        int curr = 0;
        int overlaps = 0;

        double *pos0 = &pos[3*i+0];

        for (int j=0; j<3; j++)
            index[j] = (int)((pos0[j]-BL[j])/(BU[j]-BL[j]) * size[j]);

        for (tt[0]=MAX(index[0]-1, 0); tt[0]<=MIN(index[0]+1, size[0]); tt[0]++){
        for (tt[1]=MAX(index[1]-1, 0); tt[1]<=MIN(index[1]+1, size[1]); tt[1]++){
        for (tt[2]=MAX(index[2]-1, 0); tt[2]<=MIN(index[2]+1, size[2]); tt[2]++){

            int ind = tt[0] + tt[1]*size[0] + tt[2]*size[0]*size[1];
            for (int j=0; j<count[ind]; j++){
                int tn = cells[max_particles_per_cell*ind+j];

                if (type[tn] == INACTIVE || tn == i)
                    continue;

                double *pos1 = &pos[3*tn+0];

                double dist = 0.0;
                for (int ll=0; ll<3; ll++){
                    dx[ll] = pos1[ll] - pos0[ll];
                    dist += dx[ll]*dx[ll];
                }

                if (dist < cutoffsq && curr < max_particles_per_cell){
                    double l = sqrt(dist);
                    rij[4*i*max_particles_per_cell + 4*curr + 0] = dx[0];
                    rij[4*i*max_particles_per_cell + 4*curr + 1] = dx[1];
                    rij[4*i*max_particles_per_cell + 4*curr + 2] = dx[2];
                    rij[4*i*max_particles_per_cell + 4*curr + 3] = l;
                    neighbors[i*max_particles_per_cell + curr] = tn;
                    curr++;

                    if (l < (rad[i] + rad[tn]))
                        overlaps++;
                }
            }
        }}}

        nneighs[i] = curr;
        noverlaps[i] = overlaps;
    }
}

