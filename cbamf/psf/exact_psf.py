from numpy import *
from scipy.integrate import quad
from scipy.special import jv
#^bessel function jv( nu, z)

def hellPSF_indexmatched( rho, z, alpha = 0.5):
    """
    Inputs: 
        rho: The radial distance (cylindrical coordinates) from the PSF's 
            origin, in units of 2pi/lambda (i.e. k*rho). Float scalar. 
        z: Same as above, but for the axial distance. Float scalar. 
        alpha: Float scalar on [0, pi/2]; the aperture opening of the lens. 
    """
    integrand_rl = lambda x: jv( 0, rho*sqrt(-x*(x+2)) ) * cos( z * (1+x) ) *\
        sqrt( 1 + x )
    integrand_im = lambda x: jv( 0, rho*sqrt(-x*(x+2)) ) * sin( z * (1+x) ) *\
        sqrt( 1 + x )
    sqrtP_rl = quad( integrand_rl, cos( alpha ) - 1, 0 )
    sqrtP_im = quad( integrand_im, cos( alpha ) - 1, 0 )
    return sqrtP_rl[0]**2 + sqrtP_im[0]**2 #The absolute value squared

def get_hsymm_asymm_vn( rho, z, alpha = 0.5 ):
    """
    Gets the symmetric and asymmetric portions of the vectorial PSF for an
    index-matched sample. 
    Inputs: 
        rho: The radial distance (cylindrical coordinates) from the PSF's 
            origin, in units of 2pi/lambda (i.e. k*rho). Float scalar. 
        z: Same as above, but for the axial distance. Float scalar. 
        alpha: Float scalar on [0, pi/2]; the aperture opening of the lens. 
    Outputs:
        h_symm, h_asymm: The axially symmetric and axially non-symmetric 
            portions. The actual PSF is h_symm + cos(2*phi)*h_asymm, where phi0
            is the angle of the coordinate from the light's polarization. 
    """
    #This in turn has 3 subfunctions, for integrands k1, k2, k3:
    getintegrand_k1_real = lambda theta: sin(theta) * sqrt( cos( theta ) )*\
         ( 1 - cos( theta ) ) * jv(2, rho*sin(theta) ) * cos( z*cos(theta) )
    getintegrand_k1_imag = lambda theta: sin(theta) * sqrt( cos( theta ) )*\
         ( 1 - cos( theta ) ) * jv(2, rho*sin(theta) ) * sin( z*cos(theta) )
    getintegrand_k2_real = lambda theta: sin(theta) * sqrt( cos( theta ) )*\
         ( 1 + cos( theta ) ) * jv(0, rho*sin(theta) ) * cos( z*cos(theta) )
    getintegrand_k2_imag = lambda theta: sin(theta) * sqrt( cos( theta ) )*\
         ( 1 + cos( theta ) ) * jv(0, rho*sin(theta) ) * sin( z*cos(theta) )
    getintegrand_k3_real = lambda theta: sin(theta)**2 * sqrt( cos( theta )\
         ) * jv(1, rho*sin(theta) ) * cos( z*cos(theta) )
    getintegrand_k3_imag = lambda theta: sin(theta)**2 * sqrt( cos( theta )\
         ) *jv(1, rho*sin(theta) ) * sin( z*cos(theta) )
    #Now we want to get k1,k2,k3
    k1_real = quad( getintegrand_k1_real, 0, alpha )[0]
    k1_imag = quad( getintegrand_k1_imag, 0, alpha )[0]
    k2_real = quad( getintegrand_k2_real, 0, alpha )[0]
    k2_imag = quad( getintegrand_k2_imag, 0, alpha )[0]
    k3_real = quad( getintegrand_k3_real, 0, alpha )[0]
    k3_imag = quad( getintegrand_k3_imag, 0, alpha )[0]
    h_symm = 0.25 * (k1_real**2 + k1_imag**2 + k2_real**2 + k2_imag**2) +\
        0.5 * (k3_real**2 + k3_imag**2 )
    # return h_symm
    h_asymm = 0.5 * (k3_real**2 + k3_imag**2 ) - 0.25 * (k1_real*k2_imag + \
        k1_imag * k2_real )
    return (h_symm, h_asymm )

def hellPSF_indexmatched_vector( x, y, z, alpha = 0.5 ):
    """
    """
    #1 subfunctions: to get the symmetric portion and asymmetric portion
    # def get_hasymmetric( rho, z, alpha = 0.5):
        # #again, this in turn has 3 subfunctions, for integrands k3, l1, l2
        # getintegrand_k3_real = lambda theta: sin(theta)**2 * sqrt( cos( theta )\
             # ) * jv(1, rho*sin(theta) ) * cos( z*cos(theta) )
        # getintegrand_k3_imag = lambda theta: sin(theta)**2 * sqrt( cos( theta )\
             # ) *jv(1, rho*sin(theta) ) * sin( z*cos(theta) )
        # getintegrand_l1_real = lambda theta: sqrt( cos( theta ) ) * sin(theta)*\
             # jv(0, rho*sin(theta) ) * ( 1 + cos(theta) ) * cos( z*cos(theta) )
        # getintegrand_l1_imag = lambda theta: sqrt( cos( theta ) ) * sin(theta)*\
             # jv(0, rho*sin(theta) ) * ( 1 + cos(theta) ) * sin( z*cos(theta) )
        # getintegrand_l2_real = lambda theta: sqrt( cos( theta ) ) * sin(theta)*\
             # jv(2, rho*sin(theta) ) * ( 1 - cos(theta) ) * cos( z*cos(theta) )
        # getintegrand_l2_imag = lambda theta: sqrt( cos( theta ) ) * sin(theta)*\
             # jv(2, rho*sin(theta) ) * ( 1 - cos(theta) ) * sin( z*cos(theta) )
        # #Now we get l1, l2, k3
        # k3_real = quad( getintegrand_k3_real, 0, alpha)[0]
        # k3_imag = quad( getintegrand_k3_imag, 0, alpha)[0]
        # l1_real = quad( getintegrand_l1_real, 0, alpha)[0]
        # l1_imag = quad( getintegrand_l1_imag, 0, alpha)[0]
        # l2_real = quad( getintegrand_l2_real, 0, alpha)[0]
        # l2_imag = quad( getintegrand_l2_imag, 0, alpha)[0]
        # h_asymm = 0.5 * (k3_real**2 + k3_imag**2) - 0.25 * (l1_real*l2_imag +\
            # l2_real*l1_imag)
        # return h_asymm
        #You can re-write this since l1, l2 should be ~k1, k2. Check though
        #But then get_hsymm, asymm could be 1 function that returns both components
    
    #First we get rho, phi:
    print 'Not written yet, but should be an array-accepting wrapper for ' +\
        'get_hsymm_asymm_vn()'
    return None
 
def get_hsymm_asymm_v2n_ill(rho, z, alpha = 0.5, zint = 0, n2n1 = 0.951):
    """
    Gets the symmetric and asymmetric portions of the vectorial illumination PSF
     for an index-matched sample. 
    Inputs: 
        rho: The radial distance (cylindrical coordinates) from the PSF's 
            origin, in units of 2pi/lambda (i.e. k*rho). Float scalar. 
        z: Same as above, but for the axial distance. Float scalar. 
        alpha: Float scalar on [0, pi/2]; the aperture opening of the lens. 
        zint: Float scalar. The position of the interface being imaged through. 
            Defaults to 0. 
        n2n1: Float scalar. The ratio n2/n1 of the index of refraction in the 
            second region (n2) to the first region (n1). Defaults to 0.951, or
            80:20 glycerol:water with a 1.518 refractive oil. 
        # returnDetPSF: Boolean. Set to True to return the detection PSF instead
            # of the illumination PSF. 
    Outputs:
        h_symm, h_asymm: The axially symmetric and axially non-symmetric 
            portions. The actual PSF is h_symm + cos(2*phi)*h_asymm, where phi0
            is the angle of the coordinate from the light's polarization. 
    """
    #First, since the exponent f(theta) is the same for all of them, 
    #we define that:
    n1n2 = 1.0 / n2n1 #n1 / n2 
    ftheta = lambda theta: zint * cos( theta ) - n2n1 * (zint - z) * sqrt( 1 -\
        n1n2**2 * sin(theta)**2 )
    #And the Fresnel reflectivities appear in all of them:
    taus = lambda theta: real( n1n2 * sin( 2*theta ) / sin( theta + arcsin( \
        n1n2 * sin( theta ) + 0j ) ) )
    taup = lambda theta: real( n1n2 * sin( 2*theta ) / sin( theta + arcsin( n1n2 * \
        sin( theta ) ) ) / cos(theta - arcsin( n1n2 * sin(theta ) + 0j ) ) )
    taupcos = lambda theta: taup(theta) * cos( arcsin( n1n2*sin(theta)) )
    taupsin = lambda theta: taup(theta) * n1n2 * sin(theta)
    #This in turn has 3 subfunctions, for integrands k1, k2, k3:
    getintegrand_k1_real = lambda theta: sqrt( cos(theta) ) * sin(theta ) *\
        cos( ftheta(theta) ) * 0.5 * ( taus(theta) + taupcos(theta) ) * jv(0, \
        rho*sin( theta ) )
    getintegrand_k1_imag = lambda theta: sqrt( cos(theta) ) * sin(theta ) *\
        sin( ftheta(theta) ) * 0.5 * ( taus(theta) + taupcos(theta) ) * jv(0, \
        rho*sin( theta ) )
    getintegrand_k2_real = lambda theta: sqrt( cos(theta) ) * sin(theta ) *\
        cos( ftheta(theta) ) * 0.5 * ( taus(theta) - taupcos(theta) ) * jv(2, \
        rho*sin( theta ) )
    getintegrand_k2_imag = lambda theta: sqrt( cos(theta) ) * sin(theta ) *\
        sin( ftheta(theta) ) * 0.5 * (taus(theta) - taupcos(theta) ) * jv(2, \
        rho*sin( theta ) )
    getintegrand_k3_real = lambda theta: sqrt( cos(theta) ) * sin(theta ) *\
        cos( ftheta(theta) ) * taupsin(theta) * jv(1, rho*sin( theta ) )
    getintegrand_k3_imag = lambda theta: sqrt( cos(theta) ) * sin(theta ) *\
        sin( ftheta(theta) ) * taupsin(theta) * jv(1, rho*sin( theta ) )
    #Now we want to get k1,k2,k3
    k1_real = quad( getintegrand_k1_real, 0, alpha )[0]
    k1_imag = quad( getintegrand_k1_imag, 0, alpha )[0]
    k2_real = quad( getintegrand_k2_real, 0, alpha )[0]
    k2_imag = quad( getintegrand_k2_imag, 0, alpha )[0]
    k3_real = quad( getintegrand_k3_real, 0, alpha )[0]
    k3_imag = quad( getintegrand_k3_imag, 0, alpha )[0]
    # return k1_real + 1j * k1_imag, k2_real + 1j*k2_imag, k3_real + 1j * k3_imag
    h_symm = k1_real**2 + k1_imag**2 + k2_real**2 + k2_imag**2 + 0.5 *\
        ( k3_real**2 + k3_imag**2 )
    h_asymm = 2 * ( k1_real*k2_real + k1_imag * k2_imag) + 0.5 * ( k3_real**2 +\
        k3_imag**2 )
    return (h_symm, h_asymm )

def get_hsymm_asymm_v2n_det(rho, z, alpha = 0.5, zint = 0, n2n1 = 0.951):
    """
    Gets the symmetric and asymmetric portions of the vectorial detection PSF
    for an index-matched sample. 
    Basically the same as **_ill but kills the asymmetric portion and the 
    sqrt(cos(theta)) in the integrand. 
    Inputs: 
        rho: The radial distance (cylindrical coordinates) from the PSF's 
            origin, in units of 2pi/lambda (i.e. k*rho). Float scalar. 
        z: Same as above, but for the axial distance. Float scalar. 
        alpha: Float scalar on [0, pi/2]; the aperture opening of the lens. 
        zint: Float scalar. The position of the interface being imaged through. 
            Defaults to 0. 
        n2n1: Float scalar. The ratio n2/n1 of the index of refraction in the 
            second region (n2) to the first region (n1). Defaults to 0.951, or
            80:20 glycerol:water with a 1.518 refractive oil. 
        # returnDetPSF: Boolean. Set to True to return the detection PSF instead
            # of the illumination PSF. 
    Outputs:
        h_symm, h_asymm: The axially symmetric and axially non-symmetric 
            portions. The actual PSF is h_symm + cos(2*phi)*h_asymm, where phi0
            is the angle of the coordinate from the light's polarization. 
    COMMENTS:
        If the NA of the lens is big, and the ratio n2n1 is small, then is is 
        possible to get total internal reflection in the lens for certain values
        of theta. As a result, for these angles taus, taup in the code would be 
        0 and is supposed to be no problem. However, I don't want to put if 
        statements everywhere in the code, so instead this crashes if the NA is
        too big and n2n1 is too small. What would mathematically happen with 
        the formalism we're using -- e.g. Fresnel reflectivity at the interface
        -- is that a lens with an NA that is bigger than n1/n2 will be 
        equivalent to an NA that is at n1/n2. So, for the purposes of running
        this code ad infinitum, it doesn't matter that the code crashes at 
        those points since they're included in points it doesn't crash at.
        Actually I've just made this do it imaginarily....
    """
    #First, since the exponent f(theta) is the same for all of them, 
    #we define that:
    n1n2 = 1.0 / n2n1 #n1 / n2 
    ftheta = lambda theta: zint * cos( theta ) - n2n1 * (zint - z) * sqrt( 1 -\
        n1n2**2 * sin(theta)**2 )
    #And the Fresnel reflectivities appear in all of them:
    taus = lambda theta: real( n1n2 * sin( 2*theta ) / sin( theta + arcsin( \
        n1n2 * sin( theta ) + 0j ) ) )
    taup = lambda theta: real( n1n2 * sin( 2*theta ) / sin( theta + arcsin( n1n2 * \
        sin( theta ) ) ) / cos(theta - arcsin( n1n2 * sin(theta ) + 0j ) ) )
    taupcos = lambda theta: taup(theta) * cos( arcsin( n1n2*sin(theta)) )
    taupsin = lambda theta: taup(theta) * n1n2 * sin(theta)
    #This in turn has 3 subfunctions, for integrands k1, k2, k3:
    getintegrand_k1_real = lambda theta: sin(theta ) * cos( ftheta(theta) ) *\
        0.5 * ( taus(theta) + taupcos(theta) ) * jv(0, rho*sin( theta ) )
    getintegrand_k1_imag = lambda theta: sin(theta ) * sin( ftheta(theta) ) *\
        0.5 * ( taus(theta) + taupcos(theta) ) * jv(0, rho*sin( theta ) )
    getintegrand_k2_real = lambda theta: sin(theta ) * cos( ftheta(theta) ) *\
        0.5 * ( taus(theta) - taupcos(theta) ) * jv(2, rho*sin( theta ) )
    getintegrand_k2_imag = lambda theta: sin(theta ) * sin( ftheta(theta) ) *\
        0.5 * ( taus(theta) - taupcos(theta) ) * jv(2, rho*sin( theta ) )
    getintegrand_k3_real = lambda theta: sin(theta ) * cos( ftheta(theta) ) *\
        taupsin(theta) * jv(1, rho*sin( theta ) )
    getintegrand_k3_imag = lambda theta: sin(theta ) * sin( ftheta(theta) ) *\
        taupsin(theta) * jv(1, rho*sin( theta ) )
    #Now we want to get k1,k2,k3
    k1_real = quad( getintegrand_k1_real, 0, alpha )[0]
    k1_imag = quad( getintegrand_k1_imag, 0, alpha )[0]
    k2_real = quad( getintegrand_k2_real, 0, alpha )[0]
    k2_imag = quad( getintegrand_k2_imag, 0, alpha )[0]
    k3_real = quad( getintegrand_k3_real, 0, alpha )[0]
    k3_imag = quad( getintegrand_k3_imag, 0, alpha )[0]
    h_det = k1_real**2 + k1_imag**2 + k2_real**2 + k2_imag**2 + 0.5 *\
        ( k3_real**2 + k3_imag**2 )
    return h_det

# #Sanity check:
# dum = get_hsymm_asymm_v2n_det( 0, 0, alpha = 0.5 )
# print dum
