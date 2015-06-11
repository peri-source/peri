%module fields
%{
  #define SWIG_FILE_WITH_INIT
  #include "fields.h"
  #include "util.h"
%}

%include "numpy.i"
%init %{
  import_array();
%}

%pythoncode %{
import numpy as np
dtype = np.float32
%}

%include "typemaps.i"
%numpy_typemaps(cfloat, NPY_CFLOAT, int)
/*%apply rtype *OUTPUT { rtype *t, rtype *h };*/

%apply (int DIM1, cfloat* INPLACE_ARRAY1)  {(int l, cfloat *out)};
%apply (int DIM1, cfloat* IN_ARRAY1)       {(int l, cfloat *in)};

%apply (int DIM1, float* INPLACE_ARRAY1) {(int l, float *o)};
%apply (int DIM1, float* IN_ARRAY1)      {(int l, float *iparam)};
%apply (int DIM1, double* IN_ARRAY1)      {(int P, double *x)};
%apply (int DIM1, double* IN_ARRAY1)      {(int R, double *r)};
%apply (int DIM1, double* IN_ARRAY1)      {(int P0, double *x0)};
%apply (int DIM1, double* IN_ARRAY1)      {(int R0, double *r0)};
%apply (int DIM1, float* IN_ARRAY1)      {(int l, float *i)};
%apply (int DIM1, int* IN_ARRAY1) {(int dim, int *isize)};

%include "fields.h"
