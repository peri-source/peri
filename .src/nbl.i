%module nbl 
%{
    #define SWIG_FILE_WITH_INIT
    #include "nbl.h"
%}

%include "numpy.i"
//%include "exceptions.i"
%init %{
  import_array();
%}

%pythoncode %{
import numpy as np
dtype = np.float32
%}

%include "typemaps.i"
/*%apply rtype *OUTPUT { rtype *t, rtype *h };*/

%apply (int DIM1, float* INPLACE_ARRAY1) {(int len, float *npfabric)};
%apply (int DIM1, float* INPLACE_ARRAY1) {(int len, float *npstress)};

%apply (int DIM1, float* INPLACE_ARRAY1) {(int l, float *o)};
%apply (int DIM1, float* IN_ARRAY1) {(int l, float *i)};

%apply (int DIM1, int* INPLACE_ARRAY1) {(int l, int *o)};
%apply (int DIM1, int* IN_ARRAY1) {(int l, int *i)};

%apply (int DIM1, float* IN_ARRAY1) {(int ndim, float *L)};
%apply (int DIM1, float* IN_ARRAY1) {(int ndim2, float *R)};
%apply (int DIM1, int* IN_ARRAY1)   {(int ndim3, int *pbc)};

%apply (int DIM1, double* INPLACE_ARRAY1) {(int NX, double *x)};
%apply (int DIM1, double* INPLACE_ARRAY1) {(int NR, double *r)};
%apply (int DIM1, int* INPLACE_ARRAY1)   {(int NT, int *type)};
%apply (int DIM1, float *INPLACE_ARRAY1){(int NO, float *xout)};

%include "nbl.h"
