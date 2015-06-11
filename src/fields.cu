#include <stdio.h>

#include <cufft.h>
#include <cublas.h>
#include <cuComplex.h>

#include "util.h"
#include "error.h"
#include "fields.h"

static cufftHandle g_fftplan_r2c;
static cufftHandle g_fftplan_c2r;

void f2k(float *f, cfloat *k, dimvec L, int comps);
void k2f(cfloat *k, float *f, dimvec L, int comps);

#define TILE 16
#define THREADS 512
#define BLOCKSIDE 16384
#define KERNPARMS(Ldim)                         \
    dim3 threads(THREADS, 1, 1);                \
    int __bbl = (SIZE(Ldim)-1) / THREADS + 1;   \
    dim3 blocks(__bbl % BLOCKSIDE+1, (__bbl -1) / BLOCKSIDE+1, 1); \
    //printf("threads %i blocks %i size %i\n", THREADS, SIZE(Ldim)/THREADS+1, SIZE(Ldim));

#define FPI 3.14159265358979f

#define THREAD_IDENTIFIERS(L)  \
    int _id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;    \
    int _lx = L.x;                                      \
    int _lxy = L.x * L.y;                               \
    int _bz = (_id / _lxy);                             \
    int _by = (_id - _bz*_lxy) / _lx;                   \
    int _bx = (_id - _bz*_lxy - _by*_lx) % _lx;         \
    int id = _bx + _by*_lx + _bz*_lxy;                  \
    if (id < 0 || id >= SIZE(L)) return;

#define KVECTORS(L) \
    float kx = (_bx-(_bx>L.x/2.f)*L.x)*2.f*FPI/L.x; \
    float ky = (_by-(_by>L.y/2.f)*L.y)*2.f*FPI/L.y; \
    float kz = (_bz-(_bz>L.z/2.f)*L.z)*2.f*FPI/L.z;

//=========================================================
// end of main CUDA kernels, now helper functions
//=========================================================
void __global__ _kernel_R2C(float *r, cfloat *c, dimvec L){
    THREAD_IDENTIFIERS(L);
    c[id] = init_cdata(r[id], 0.0f);
}

void __global__ _kernel_C2R(cfloat *c, float *r, dimvec L){
    THREAD_IDENTIFIERS(L);
    r[id] = c[id].x;
}

void R2C(float *r, cfloat *c, dimvec L, int comps){
    KERNPARMS(L)
    _kernel_R2C<<<blocks, threads>>>(r, c, L);
    cudaThreadSynchronize();
}

void C2R(cfloat *c, float *r, dimvec L, int comps){
    KERNPARMS(L)
    _kernel_C2R<<<blocks, threads>>>(c, r, L);
    cudaThreadSynchronize();
}

void f2k(float *f, cfloat *k, dimvec L, int comps){
    cfloat *in = createDeviceArrayC(SIZE(L));
    R2C(f, in, L, comps);
    fft_r2c(g_fftplan_r2c, (fft_float_c*)(k), (fft_float_c*)(k), CUFFT_FORWARD);
    freeDeviceArrayC(in);
}

void k2f(cfloat *k, float *f, dimvec L, int comps){
    cfloat *out = createDeviceArrayC(SIZE(L));
    fft_c2r(g_fftplan_c2r, (fft_float_c*)(k), (fft_float_c*)(out), CUFFT_INVERSE);
    C2R(out, f, L, comps);
    freeDeviceArrayC(out);
}

dimvec arr2dimvec(int dim, int *i){
    dimvec size;
    size.x = i[0];
    size.y = i[1];
    size.z = (dim == 3) ? i[2] : 1;
    return size;
}

int *dimvec2arr(dimvec i){
    int *o = (int*)malloc(sizeof(int)*3);
    o[0] = i.x;
    o[1] = i.y;
    o[2] = i.z;
    return o;
}

//========================================================
// finally, the swig wrapper functions
//========================================================
field *createField(int dim, int *isize){
    dimvec size = arr2dimvec(dim, isize);

    int len = SIZE(size);
    float *arr = (float*)malloc(sizeof(float)*len);

    field ff;
    ff.arr = arr;
    ff.dim = dim;
    ff.len = len;
    ff.size = size;

    field *out = (field*)malloc(sizeof(field));
    memcpy(out, &ff, sizeof(field));
    return out;
}

cfield *createCField(int dim, int *isize){
    dimvec size = arr2dimvec(dim, isize);

    int len = SIZE(size);
    cfloat *arr = (cfloat*)malloc(sizeof(cfloat)*len);

    cfield ff;
    ff.arr = arr;
    ff.dim = dim;
    ff.len = len;
    ff.size = size;

    cfield *out = (cfield*)malloc(sizeof(cfield));
    memcpy(out, &ff, sizeof(cfield));
    return out;
}

field *createFieldGPU(field *cpu){
    float *arr = createDeviceArray(SIZE(cpu->size));

    field *out = (field*)malloc(sizeof(field));
    memcpy(out, cpu, sizeof(field));
    out->arr = arr;
    return out;
}

cfield *createCFieldGPU(field *cpu){
    cfloat *arr = createDeviceArrayC(SIZE(cpu->size));

    cfield *out = (cfield*)malloc(sizeof(cfield));
    memcpy(out, cpu, sizeof(cfield));
    out->arr = arr;
    return out;
}

void field_cpu2gpu(field *cpu, field *gpu){
    copyToDevice(SIZE(cpu->size), cpu->arr, gpu->arr);
    gpu->dim = cpu->dim;
    gpu->len = cpu->len;
    gpu->size = cpu->size;
}

void field_gpu2cpu(field *gpu, field *cpu){
    cpu->dim = gpu->dim;
    cpu->len = gpu->len;
    cpu->size = gpu->size;
    copyFromDevice(gpu->arr, SIZE(cpu->size), cpu->arr);
}

void freeField(field *f){
    free(f->arr);
    free(f);
}

void freeFieldGPU(field *f){
    freeDeviceArray(f->arr);
    free(f);
}

void freeCFieldGPU(cfield *f){
    freeDeviceArrayC(f->arr);
    free(f);
}

cfield *field_deepcopy(cfield *f){
    cfloat *arr = createDeviceArrayC(SIZE(f->size));

    cfield *out = (cfield*)malloc(sizeof(cfield));
    memcpy(out, f, sizeof(cfield));
    out->arr = arr;
    CSAFECALL( cudaMemcpy(out->arr, f->arr, sizeof(cfloat)*SIZE(f->size), cudaMemcpyDeviceToDevice) );
    return out;
}

void fieldGet(field *f, int l, float *out){
    if (l != f->len)
        printf("Wrong size, must be %i long\n", f->len);
    copyArray(f->arr, l, out);
}

void fieldSet(field *f, int l, float *in){
    if (l != f->len)
        printf("Wrong size, must be %i long\n", f->len);
    copyArray(in, l, f->arr);
}

void setupFFT(field *f){
    dimvec L = f->size;
    cufftPlan3d(&g_fftplan_r2c, L.z, L.y, L.x, FORWARD_FFT);
    cufftPlan3d(&g_fftplan_c2r, L.z, L.y, L.x, BACKWARD_FFT);
    cublasInit();
}

void destroyFFT(){
    cufftDestroy(g_fftplan_r2c);
    cufftDestroy(g_fftplan_c2r);
}


//============================================================================
// device functions that create platonic images of spheres (circles)
//============================================================================
__device__ float __dot(float *a, float *b){
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ float __volume1(float r){return 2.0f*r; }
__device__ float __volume2(float r){return FPI*r*r; }
__device__ float __volume3(float r){return 4.0f*FPI/3.0f*r*r*r; }
__device__ float __ksphere1(float r){return (sin(r)/r);}
__device__ float __ksphere2(float r){return 2.0f*j1(r)/r;}
__device__ float __ksphere3(float r){float d=1.0f/r; return 3.0f*(sin(r)*d-cos(r))*d*d;}

__device__ cfloat __create_particle(float *xin, float rin, float *kv,
        float k, int id, float norm){
    cfloat out = init_cdata(0,0);
    float r = rin;
    float d = __dot(kv, xin);

    float smoothing = expf(-k*k / (FPI*FPI/2));
    cfloat texp = init_cdata(cosf(-d), sinf(-d));
    out.x = 1.0f*(id==0) + __ksphere3(k*r)*(id!=0);
    out.x = out.x * __volume3(r) * smoothing * norm;
    out = cmul(out, texp);
    return out;
}

__global__ void _particles_to_ccd(cfloat *ccd, dimvec L, int N, double *x, double *r){
    THREAD_IDENTIFIERS(L);
    KVECTORS(L);

    cfloat lccd = init_cdata(0,0);
    float kv[3] = {kx, ky, kz};
    float k = sqrt(__dot(kv, kv));
    float fx[3] = {0,0,0};

    float norm = 1.0f/SIZE(L);
    for (int i=0; i<N; i++){
        fx[0] = x[3*i+0];
        fx[1] = x[3*i+1];
        fx[2] = x[3*i+2];
        cfloat tmp = __create_particle(fx, (float)r[i], kv, k, id, norm);
        lccd = cadd(lccd, tmp);
    }

    ccd[id] = lccd;
}

__global__ void _update_particles_to_ccd(cfloat *ccd, dimvec L, int N, double *x0, double *r0,
        double *x, double *r){
    THREAD_IDENTIFIERS(L);
    KVECTORS(L);

    cfloat lccd = init_cdata(0,0);
    float kv[3] = {kx, ky, kz};
    float k = sqrt(__dot(kv, kv));
    float fx0[3] = {0,0,0};
    float fx1[3] = {0,0,0};

    cfloat before = ccd[id];

    float norm = 1.0f/SIZE(L);
    for (int i=0; i<N; i++){
        fx0[0] = x0[3*i+0];
        fx0[1] = x0[3*i+1];
        fx0[2] = x0[3*i+2];
        fx1[0] = x[3*i+0];
        fx1[1] = x[3*i+1];
        fx1[2] = x[3*i+2];

        cfloat tmp = __create_particle(fx0, (float)r0[i], kv, k, id, norm);
        lccd = csub(lccd, tmp);

        tmp = __create_particle(fx1, (float)r[i], kv, k, id, norm);
        lccd = cadd(lccd, tmp);
    }

    ccd[id] = cadd(before, lccd);
}

//============================================================================
// device functions to create normalized point spread functions
//============================================================================
__device__ float __psf_pade_evaluate(float x, float *params, int PI, int PJ){
    float num = 0.0f, den = 0.0f;
    for (int i=(PI-1);    i>=0;  i--) num = num*x + params[i];
    for (int i=(PI+PJ-1); i>=PI; i--) den = den*x + params[i];
    float v = (1.0f + num*x)/(1.0f + den*x);
    return v;
}

__device__ float __psf_sigmoid_evaluate(float x, float *params){
    return (1.0f + expf(-params[1]*params[0])) / (1.0f + expf(params[1]*(x - params[0])));
}

#define PADE_FUNCTION_ORDER(PI,PJ) \
__global__ void _kcreate_psf_isotropic_pade_##PI##_##PJ(cfloat *psf,    \
        dimvec L, float *params){                                       \
    THREAD_IDENTIFIERS(L);                                              \
    KVECTORS(L);                                                        \
    float k = (kx*kx + ky*ky + kz*kz)*params[PI+PJ];                    \
    float v = __psf_pade_evaluate(k, params, PI, PJ);                   \
    psf[id] = init_cdata(v, 0);                                         \
}

PADE_FUNCTION_ORDER(2,4);
PADE_FUNCTION_ORDER(3,5);
PADE_FUNCTION_ORDER(3,7);
PADE_FUNCTION_ORDER(7,11);

__global__ void _kcreate_psf_gaussian(cfloat *psf, dimvec L, float *params){
    THREAD_IDENTIFIERS(L);
    KVECTORS(L);

    float sig = params[0];
    float ksq = (kx*kx + ky*ky + kz*kz);
    float exx = expf(-ksq*(sig*sig)/4);
    psf[id] = init_cdata(exx, 0);
}

__global__ void _kcreate_psf_isotropic_disc(cfloat *psf, dimvec L, float *params){
    THREAD_IDENTIFIERS(L);
    KVECTORS(L);

    float k = sqrt(kx*kx + ky*ky + kz*kz);
    float v = __psf_sigmoid_evaluate(k, params);
    psf[id] = init_cdata(v, 0);
}

__global__ void _kcreate_psf_xy_disc_z_pade(cfloat *psf, dimvec L, float *params){
    THREAD_IDENTIFIERS(L);
    KVECTORS(L);

    float k = sqrt(kx*kx + ky*ky);
    float Z = sqrt(kz*kz);
    float v1 = __psf_sigmoid_evaluate(k, params);
    float v2 = __psf_pade_evaluate(Z, &params[2], 3, 5);
    psf[id] = init_cdata(v1*v2, 0);
}

__global__ void _apply_psf(cfloat *kccd, cfloat *kpsf, dimvec L){
    THREAD_IDENTIFIERS(L);

    cfloat norm = init_cdata(1.0f/kpsf[0].x, 0);
    kccd[id] = cmul(norm, cmul(kccd[id], kpsf[id]));
}

#define PADE_LOOPER(PI,PJ) \
if (flags & PSF_ISOTROPIC_PADE_##PI##_##PJ) \
    _kcreate_psf_isotropic_pade_##PI##_##PJ<<<blocks, threads>>>(kpsf, kccd->size, param);

void apply_psf(cfield *kccd, int l, float *iparam, int flags){
    KERNPARMS(kccd->size);

    float *param = createDeviceArray(l);
    cfloat *kpsf = createDeviceArrayC(SIZE(kccd->size));

    if (l > 0) copyToDevice(l, iparam, param);

    if (flags & PSF_ISOTROPIC_GAUSSIAN)
        _kcreate_psf_gaussian<<<blocks, threads>>>(kpsf, kccd->size, param);
    if (flags & PSF_ISOTROPIC_DISC)
        _kcreate_psf_isotropic_disc<<<blocks, threads>>>(kpsf, kccd->size, param);
    if (flags & PSF_XY_DISC_Z_PADE)
        _kcreate_psf_xy_disc_z_pade<<<blocks, threads>>>(kpsf, kccd->size, param);
    PADE_LOOPER(2,4);
    PADE_LOOPER(3,5);
    PADE_LOOPER(3,7);
    PADE_LOOPER(7,11);

    KERNCHECK( _apply_psf<<<blocks, threads>>>(kccd->arr, kpsf, kccd->size) );

    freeDeviceArray(param);
    freeDeviceArrayC(kpsf);

error:
    return;
}

//============================================================================
//============================================================================
__global__ void _trim_field(float *big, float *small, dimvec Lb, dimvec Ls, int PAD){
    THREAD_IDENTIFIERS(Lb);
    int lsx = Ls.x;
    int lsy = Ls.y;
    int lsz = Ls.z;

    float lf = big[id];
    if (_bx < lsx && _by < lsy && _bz < lsz){
        int oid = _bx + _by*lsx + _bz*lsx*lsy;
        small[oid] = lf;
    }
}

void trim_field(field *big, field *small, int PAD){
    KERNPARMS(big->size);
    _trim_field<<<blocks, threads>>>(big->arr, small->arr, big->size, small->size, PAD);
}

field *create_pad_image_gpu(field *f, int pad){
    dimvec dold = f->size;
    dimvec dnew = dold;
    dnew.x += pad; dnew.y += pad; dnew.z += pad;

    f->size = dnew;
    field *o = createFieldGPU(f);
    f->size = dold;

    return o;
}

cfield *create_padded_ccd(int dim, int *isize, int pad){
    field *f = createField(dim, isize);

    const int PAD = pad;
    dimvec dold = f->size;
    dimvec dnew = dold;
    dnew.x += PAD; dnew.y += PAD; dnew.z += PAD;

    f->size = dnew;
    cfield *o = createCFieldGPU(f);
    f->size = dold;

    freeField(f);
    return o;
}

//============================================================================
//============================================================================
cfield *py_cfloat2cfield(int l, cfloat *in, int dim, int *isize){
    dimvec size = arr2dimvec(dim, isize);

    int len = SIZE(size);
    cfloat *arr = createDeviceArrayC(len);
    copyToDeviceC(l, in, arr);

    cfield ff;
    ff.arr = arr;
    ff.dim = dim;
    ff.len = len;
    ff.size = size;

    cfield *out = (cfield*)malloc(sizeof(cfield));
    memcpy(out, &ff, sizeof(cfield));
    return out;
}

void py_cfield2cfloat(cfield *in, int l, cfloat *out){
    copyFromDeviceC(in->arr, l, out);
}

void create_kspace_spheres(cfield *kccd, int N3, double *x, int N, double *r){
    if (N3 != 3*N) log_err("Positions and radii are not the same size\n");

    double *dx = createDeviceArrayD(N3);
    double *dr = createDeviceArrayD(N);
    copyToDeviceD(N3, x, dx);
    copyToDeviceD(N,  r, dr);

    KERNPARMS(kccd->size);
    KERNCHECK( _particles_to_ccd<<<blocks, threads>>>(kccd->arr, kccd->size, N, dx, dr) );

    freeDeviceArrayD(dx);
    freeDeviceArrayD(dr);

error:
    return;
}

void update_kspace_spheres(cfield *kccd, int P0, double *x0, int R0, double *r0,
        int P, double *x, int R, double *r){
    double *dx0 = createDeviceArrayD(P0);
    double *dr0 = createDeviceArrayD(R0);
    double *dx = createDeviceArrayD(P);
    double *dr = createDeviceArrayD(R);
    if (P0 != P || R0 != R || P0 != 3*R0) log_err("Sizes are not consistent\n");

    copyToDeviceD(P0, x0, dx0);
    copyToDeviceD(R0, r0, dr0);
    copyToDeviceD(P,  x,  dx);
    copyToDeviceD(R,  r,  dr);

    KERNPARMS(kccd->size);
    KERNCHECK( _update_particles_to_ccd<<<blocks, threads>>>(kccd->arr, kccd->size, R, dx0, dr0, dx, dr) );

    freeDeviceArrayD(dx0);
    freeDeviceArrayD(dr0);
    freeDeviceArrayD(dx);
    freeDeviceArrayD(dr);

error:
    return;
}

image *create_image_package(field *out, int pad){
    const int PAD = pad;
    field *dbig = create_pad_image_gpu(out, PAD);
    field *dsmall = createFieldGPU(out);
    cfield *copy = createCFieldGPU(dbig);
    setupFFT(dbig);

    image *o = (image*)malloc(sizeof(image));
    o->dbig = dbig;
    o->dsmall = dsmall;
    o->copy = copy;
    return o;
}

void free_image_package(image *im){
    freeFieldGPU(im->dsmall);
    freeFieldGPU(im->dbig);
    freeCFieldGPU(im->copy);
    destroyFFT();
}

__global__ void _produce_darkfield_image(float *bkg, float *platonic, dimvec L){
    THREAD_IDENTIFIERS(L);
    platonic[id] = bkg[id] * (1 - platonic[id]);
}

void update_image_with_bkg(cfield *kccd, field *bkg, field *out, image *im, int l, float *iparam, int flags, int pad){
    /* 
       This function intends to calculate PSF (*) ( BKG (1 - PLATONIC) ) image.
       to do so, we must calculate 
            y = IFFT(PLATONIC)
            q = BKG * (1 - PLATONIC)
            out = IFFT( FFT(PSF) * FFT(q).conj() )
    */
    const int PAD = pad;
    CSAFECALL(
        cudaMemcpy(im->copy->arr, kccd->arr, sizeof(cfloat)*SIZE(kccd->size),
            cudaMemcpyDeviceToDevice)
    );

    KERNPARMS(kccd->size);

    // must ifft the kccd field into a platonic image from (0,1)
    KERNCHECK( k2f(kccd->arr, im->dbig->arr, im->dbig->size, 1) );

    // multiply bkg * (1-ccd), save to dbig
    KERNCHECK( _produce_darkfield_image<<<blocks, threads>>>(bkg->arr, im->dbig->arr, kccd->size) );

    // and take fft, save to copy
    KERNCHECK( f2k(im->dbig->arr, im->copy->arr, im->dbig->size, 1) );

    // apply PSF and inverse fft
    apply_psf(im->copy, l, iparam, flags);

    // trim down to size
    KERNCHECK( k2f(im->copy->arr, im->dbig->arr, im->dbig->size, 1) );

    trim_field(im->dbig, im->dsmall, PAD);
    field_gpu2cpu(im->dsmall, out);
error:
    return;
}

void update_image(cfield *kccd, field *out, image *im, int l, float *iparam, int flags, int pad){
    const int PAD = pad;
    CSAFECALL(
        cudaMemcpy(im->copy->arr, kccd->arr, sizeof(cfloat)*SIZE(kccd->size),
            cudaMemcpyDeviceToDevice)
    );
    apply_psf(im->copy, l, iparam, flags);
    KERNCHECK( k2f(im->copy->arr, im->dbig->arr, im->dbig->size, 1) );

    trim_field(im->dbig, im->dsmall, PAD);
    field_gpu2cpu(im->dsmall, out);
error:
    return;
}

void process_image(field *f, int N3, double *x, int N, double *r,
        int l, float *iparam, int flags, int pad){

    const int PAD = pad;
    field *dbig = create_pad_image_gpu(f, PAD);
    field *dsmall = createFieldGPU(f);
    cfield *kccd = createCFieldGPU(dbig);
    setupFFT(dbig);

    create_kspace_spheres(kccd, N3, x, N, r);
    apply_psf(kccd, l, iparam, flags);
    KERNCHECK( k2f(kccd->arr, dbig->arr, dbig->size, 1) );

    trim_field(dbig, dsmall, PAD);
    field_gpu2cpu(dsmall, f);

    destroyFFT();
    freeCFieldGPU(kccd);
    freeFieldGPU(dbig);
    freeFieldGPU(dsmall);

error:
    return;
}

#ifdef TEST2

int main(){
    const int GPAD = 20;
    ran_seed(5);
    int size[3] = {32,32,32};
    //int size[3] = {71,71,21};
    int N = 6;
    float L = 10;
    float *s = (float*)malloc(sizeof(float)*3*N);
    float *r = (float*)malloc(sizeof(float)*N);
    float *p = (float*)malloc(sizeof(float)*2);

    for (int i=0; i<N; i++){
        s[3*i+0] = L*ran_ran2();
        s[3*i+1] = L*ran_ran2();
        s[3*i+2] = L*ran_ran2();
        r[i] = 10;
    }
    field *f = createField(3, size);
    for (int i=0; i<1; i++)
        process_image(f, 3*N, s, N, r, 2, p, PSF_ISOTROPIC_DISC, GPAD);

    float x0[3] = {s[0], s[1], s[2]};
    float x1[3] = {s[0]+1, s[1], s[2]};
    float r0[1] = {r[0]};
    float r1[1] = {r[0]};

    image *im = create_image_package(f, GPAD);
    cfield *kccd = create_padded_ccd(3, size, GPAD);
    create_kspace_spheres(kccd, 3*N, s, N, r);
    update_kspace_spheres(kccd, 3, x0, 1, r0, 3, x1, 1, r1);
    update_image(kccd, f, im, 2, p, PSF_ISOTROPIC_DISC, GPAD);

    free(s);
    free(r);
    free(p);
    freeField(f);
    freeCFieldGPU(kccd);
}
#endif

/*===============================================================
  the following function comes from Mark Harris @ NVIDIA
  from a talk Optimizing Parallel Reduction in CUDA
  along with substantial logic from
  http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=136807&file=%5CSpikingNeurons(ModelDB)%5Creduction.cu
  ===============================================================*/
/*#define MAX_REDUCTION_THREADS 256
#define MAX_REDUCTION_BLOCKS 64

__global__ void freduce(float *g_idata, float *g_odata, int n, int blockSize)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;

    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// we assume n in a power of 2!!
void getReductionBlocksAndThreads(int n, int *blocks, int *threads){
    *threads = (n < MAX_REDUCTION_THREADS*2) ? n*2 : MAX_REDUCTION_THREADS;
    *blocks = (n + (*threads * 2 - 1)) / (*threads * 2);
    *blocks = MIN(MAX_REDUCTION_BLOCKS, *blocks);
}

float reduceSum(float *in, dimvec *L, int comps){
    float host_out = 0.0;
    float *dev_out = createDeviceArray(L, comps);

    int n = SIZE(*L);
    int threads = 0;
    int blocks  = 0;
    getReductionBlocksAndThreads(n, &blocks, &threads);

    int smemSize = (threads<=32)?2*threads*sizeof(float) : threads*sizeof(float);
    freduce<<< blocks, threads, smemSize >>>(in, dev_out, n, threads);

    int s=blocks;
    while(s > 1) {
        getReductionBlocksAndThreads(s, &blocks, &threads);
        smemSize = (threads<=32)?2*threads*sizeof(float) : threads*sizeof(float);
        freduce<<< blocks, threads, smemSize >>>(dev_out, dev_out, n, threads);
        s = (s + (threads*2-1)) / (threads*2);
    }

    CUDA_SAFE_CALL(cudaMemcpy(&host_out, dev_out, sizeof(float), cudaMemcpyDeviceToHost));
    freeDeviceArray(dev_out);

    return host_out;
}*/
