import numpy as np

try:
    from scipy.weave import inline
except ImportError as e:
    try:
        from weave import inline
    except ImportError as e:
        pass

functions = r"""
double PI = 3.1415926535;

double sgn(double x){
    return (x > 0) - (x < 0);
}

/*
  To use this function, you must provide:
    * N -- the number of values in the table
    * table -- the values for each part
    * dl -- domain left value
    * dr -- domain right value
*/
double periodic_lookup(double x, double *table, double dl, double dr, int N){
    int r = (int)((x - dl) / (dr - dl));
    double dx = (dr - dl)/N;
    double xr = x - (dr - dl)*r;

    int i = (int)(xr / dx); 
    int j = (i+1)*(i <= N-2);
    double v0 = table[i];
    double v1 = table[j];

    double d = (xr - i*dx)/dx;
    return v0 + (v1-v0)*d;
}

double fast_erf(double x){
    double sgn = 1.0;

    if (x < 0){
        sgn = -1.0;
        x = -x;
    }

    double p = 0.47047;
    double a1 =  0.3480242;
    double a2 = -0.0958798;
    double a3 =  0.7478556;
    double t1 = 1.0/(1 + p*x);
    double t2 = t1*t1;
    double t3 = t1*t2;
    return sgn*(1 - (a1*t1 + a2*t2 + a3*t3)*exp(-x*x));
}

static union
{
    double d;
    struct
    {
        #ifdef LITTLE_ENDIAN
            int j, i;
        #else
            int i, j;
        #endif
    } n;
} _eco;

#define EXP_A (1048576 /M_LN2)
#define EXP_C 60801
#define fast_exp(y) (_eco.n.i = EXP_A*(y) + (1072693248 - EXP_C), _eco.d)

double fast_j0(double x){
    double pi  = 3.14159265;
    double pi4 = pi/4;

    double a0 =  0.99999990;
    double a1 = -2.24999239;
    double a2 =  1.26553572;
    double a3 = -0.31602189;
    double a4 =  0.04374224;
    double a5 = -0.00331563;

    double b0 =  0.79788454;
    double b1 = -0.00553897;
    double b2 =  0.00099336;
    double b3 = -0.00044346; 
    double b4 =  0.00020445;
    double b5 = -0.00004959;

    double c0 = -0.04166592;
    double c1 =  0.00239399;
    double c2 = -0.00073984;
    double c3 =  0.00031099;
    double c4 = -0.00007605;

    double out = 0.0;
    if (x < 3){
        double x3 = x*x/9;
        out = a0 + x3*(a1 + x3*(a2 + x3*(a3 + x3*(a4 + x3*a5))));
    } else {
        double x3 = 3/x;
        double x6 = x3*x3;
        double f = b0 + x6*(b1 + x6*(b2 + x6*(b3 + x6*(b4 + x6*b5))));
        double t = x - pi4 + x3*(c0 + x6*(c1 + x6*(c2 + x6*(c3 + x6*c4))));
        out = 1.0/sqrt(x)*f*cos(t);
    }
    return out;
}

double fast_j1(double x){
    double pi  = 3.14159265;
    double pi4 = 3*pi/4;

    double a0 =  0.50000000;
    double a1 = -0.56249945;
    double a2 =  0.21093101;
    double a3 = -0.03952287;
    double a4 =  0.00439494;
    double a5 = -0.00028397;

    double b0 =  0.79788459;
    double b1 =  0.01662008;
    double b2 = -0.00187002;
    double b3 =  0.00068519;
    double b4 = -0.00029440;
    double b5 =  0.00006952;

    double c0 =  0.12499895;
    double c1 = -0.00605240;
    double c2 =  0.00135825;
    double c3 = -0.00049616;
    double c4 =  0.00011531;

    double out = 0.0;
    if (x < 3){
        double x3 = x*x/9;
        out = x*(a0 + x3*(a1 + x3*(a2 + x3*(a3 + x3*(a4 + x3*a5)))));
    } else {
        double x3 = 3/x;
        double x6 = x3*x3;
        double f = b0 + x6*(b1 + x6*(b2 + x6*(b3 + x6*(b4 + x6*b5))));
        double t = x - pi4 + x3*(c0 + x6*(c1 + x6*(c2 + x6*(c3 + x6*c4))));
        out = 1.0/sqrt(x)*f*cos(t);
    }
    return out;
}
"""

def build_table(func, N):
    x = np.linspace(0, 2*np.pi, N)
    t = func(x)
    return [N, t, 0, 2*np.pi]

def _eval_table(x, table):
    code = """
    for (int i=0; i<N; i++){
     out[i]=periodic_lookup(x[i], _table, _dl, _dr, _N);
    }
    """
    N = x.ravel().shape[0]
    xin = x.ravel()
    out = 0*x

    _N, _table, _dl, _dr = table
    inline(code, arg_names=['x', 'N', 'out', '_table', '_dl', '_dr', '_N'],
            verbose=0, support_code=functions)
    return out.reshape(x.shape)

def fast_j0(x):
    code = """
    for (int i=0; i<N; i++){
       out[i] = fast_j0(x[i]);
    }
    """
    N = x.ravel().shape[0]
    xin = x.ravel()
    out = 0*x

    inline(code, arg_names=['x', 'N', 'out'], verbose=0, support_code=functions)
    return out.reshape(x.shape)

def fast_j1(x):
    code = """
    for (int i=0; i<N; i++){
      out[i] = fast_j1(x[i]);
    }
    """
    N = x.ravel().shape[0]
    xin = x.ravel()
    out = 0*x

    inline(code, arg_names=['x', 'N', 'out'], verbose=0, support_code=functions)
    return out.reshape(x.shape)

def fast_j2(x):
    return 2./(x+1e-15)*fast_j1(x) - fast_j0(x)

