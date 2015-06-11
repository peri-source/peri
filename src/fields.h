#ifndef __FIELDS_H__
#define __FIELDS_H__

#include "types.h"

#define init_cdata(x,y) make_cuFloatComplex(x,y)
#define FORWARD_FFT CUFFT_C2C
#define BACKWARD_FFT CUFFT_C2C
#define fft_r2c cufftExecC2C
#define fft_c2r cufftExecC2C
#define fft_float_r cufftReal
#define fft_float_c cufftComplex
#define Iamax cublasIsamax
#define Iamin cublasIsamin
#define axpy cublasSaxpy
#define vec_copy cublasScopy
#define cadd cuCaddf
#define csub cuCsubf
#define cmul cuCmulf

#define SIZE(s) (s.x*s.y*s.z)

#define PSF_NONE                (1 << 1)
#define PSF_ISOTROPIC_GAUSSIAN  (1 << 2)
#define PSF_ISOTROPIC_PADE_2_4  (1 << 3)
#define PSF_ISOTROPIC_PADE_3_5  (1 << 4)
#define PSF_ISOTROPIC_PADE_3_7  (1 << 5)
#define PSF_ISOTROPIC_PADE_7_11 (1 << 6)
#define PSF_ISOTROPIC_DISC      (1 << 7)
#define PSF_XY_DISC_Z_PADE      (1 << 8)

typedef struct {
    field *dbig;
    field *dsmall;
    cfield *copy;
} image;

image *create_image_package(field *out, int pad);
void free_image_package(image *im);

//====================================================
field *createField(int dim, int *isize);
field *createFieldGPU(field *cpu);
void field_cpu2gpu(field *cpu, field *gpu);
void field_gpu2cpu(field *gpu, field *cpu);
void freeField(field *f);
void freeFieldGPU(field *f);
void freeCFieldGPU(cfield *f);

void fieldGet(field *f, int l, float *o);
void fieldSet(field *f, int l, float *i);

void setupFFT(field *f);
void process_image(field *f, int P, double *x, int R, double *r, int l, float *iparam, int flags, int pad);

/*float reduceMax(float *in, dimvec *L, int comps);
float reduceMin(float *in, dimvec *L, int comps);
float reduceSum(float *in, dimvec *L, int comps);*/

cfield *create_padded_ccd(int dim, int *isize, int pad);
cfield *field_deepcopy(cfield *f);
void create_kspace_spheres(cfield *kccd,
        int P,  double *x,  int R, double *r);
void update_kspace_spheres(cfield *kccd,
        int P0, double *x0, int R0, double *r0,
        int P,  double *x,  int R,  double *r);
void update_image(cfield *kccd, field *out, image *im,
        int l, float *iparam, int flags, int pad);

cfield *py_cfloat2cfield(int l, cfloat *in, int dim, int *isize);
void py_cfield2cfloat(cfield *in, int l, cfloat *out);


#endif
