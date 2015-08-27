"""
Calculates PSFs with better coding practice
"""

from numpy import *
from scipy.special import jv
from scipy.integrate import quad

def f_theta( theta, zint, z, n2n1 = 0.95 ):
    """
    """
    return zint * cos(theta) - n2n1*(zint-z)* sqrt( 1 - (sin(theta)/n2n1)**2)
    
def get_get_taus( n1n2=1.05 ):
    get_taus = lambda theta: n1n2*sin(2*theta)/sin(theta+arcsin( n1n2*sin(theta)))
    return get_taus

def get_get_taup( n1n2=1.05):
    get_taup = lambda theta: n1n2*sin(2*theta)/sin(theta+arcsin( n1n2*sin(theta)))/\
        cos(theta - arcsin(n1n2*sin(theta)))
    return get_taup

def get_get_Kprefactor( rho,z, alpha = 1.0, zint = 100.0, n2n1 = 0.95, get_hdet=False):
    if get_hdet:
        to_return_rl = lambda theta: sin(theta)*cos( f_theta( theta, zint,z,\
            n2n1=n2n1))
        to_return_im = lambda theta: sin(theta)*sin( f_theta( theta, zint,z,\
            n2n1=n2n1))
    else:
        to_return_rl = lambda theta: sin(theta)*cos( f_theta( theta, zint,z,\
            n2n1=n2n1)) * sqrt(cos(theta))
        to_return_im = lambda theta: sin(theta)*sin( f_theta( theta, zint,z,\
            n2n1=n2n1)) * sqrt(cos(theta))
    return to_return_rl, to_return_im
    
def get_K( rho, z, alpha=1.0, zint=100.0, n2n1=0.95, get_hdet=False, K=1):
    """
    """
    n1n2 = 1.0/n2n1
    
    get_Kprefactor = get_get_Kprefactor(rho, z, alpha=alpha, zint=zint, n2n1=\
        n2n1,get_hdet=get_hdet)
    get_taus = get_get_taus( n1n2=n1n2)
    get_taup = get_get_taup( n1n2=n1n2)
    
    if K==1:
        integrand_rl = lambda theta: get_Kprefactor[0](theta)*0.5*( get_taus(\
            theta) + get_taup(theta)*cos(arcsin(n1n2*sin(theta))) )*jv(0, rho*sin(theta))
        integrand_im = lambda theta: get_Kprefactor[1](theta)*0.5*( get_taus(\
            theta) + get_taup(theta)*cos(arcsin(n1n2*sin(theta))) )*jv(0, rho*sin(theta))
    elif K==2:
        integrand_rl = lambda theta: get_Kprefactor[0](theta)*0.5*( get_taus(\
            theta) - get_taup(theta)*cos(arcsin(n1n2*sin(theta))) )*jv(2, rho*sin(theta))
        integrand_im = lambda theta: get_Kprefactor[1](theta)*0.5*( get_taus(\
            theta) - get_taup(theta)*cos(arcsin(n1n2*sin(theta))) )*jv(2, rho*sin(theta))
    elif K==3:
        integrand_rl = lambda theta: get_Kprefactor[0](theta)*get_taup(theta)*\
            sin(theta)*n1n2*jv(1, rho*sin(theta))
        integrand_im = lambda theta: get_Kprefactor[1](theta)*get_taup(theta)*\
            sin(theta)*n1n2*jv(1, rho*sin(theta))
    else:
        raise RuntimeError
    kint_rl = quad( integrand_rl, 0, alpha)[0]
    kint_im = quad( integrand_im, 0, alpha)[0]
    return kint_rl, kint_im

def get_hsym_asym( rho, z, alpha=1.0, zint=100.0, n2n1=0.95, get_hdet=False):
    
    K1 = get_K( rho, z, K=1, alpha=alpha,zint=zint,n2n1=n2n1,get_hdet=get_hdet)
    K2 = get_K( rho, z, K=2, alpha=alpha,zint=zint,n2n1=n2n1,get_hdet=get_hdet)
    K3 = get_K( rho, z, K=3, alpha=alpha,zint=zint,n2n1=n2n1,get_hdet=get_hdet)
    
    hsym = K1[0]**2+K1[1]**2 + K2[0]**2+K2[1]**2 + 0.5*(K3[0]**2+K3[1]**2)
    hasym= 2*(K1[0]*K2[0] + K1[1]*K2[1]) + 0.5*(K3[0]**2+K3[1]**2)
    
    return hsym,hasym
    
