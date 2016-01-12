import numpy as np
from scipy.special import j0,j1

from cbamf.comp.psfs import PSF

def j2(x):
    """ A fast j2 defined in terms of other special functions """
    to_return = 2./x*j1(x) - j0(x)
    to_return[x==0] = 0
    return to_return

"""
Global magic #'s for a good number of points for gauss-legendre quadrature. 20
gives double precision (no difference between 20 and 30 and doing all the
integrals with scipy.quad). The integrals are only over the acceptance angle of
the lens, so they shouldn't be too rapidly varying, but you might need more
points for large z,zint (large compared to 100).
"""
NPTS = 20
PTS,WTS = np.polynomial.legendre.leggauss(NPTS)
PTS_HG,WTS_HG = np.polynomial.hermite.hermgauss(NPTS*2)
PTS_HG = PTS_HG[NPTS:]
WTS_HG = WTS_HG[NPTS:]*np.exp(PTS_HG*PTS_HG)

def f_theta(cos_theta, zint, z, n2n1 = 0.95):
    """
    """
    # return zint * cos_theta - (zint-z)*np.sqrt( n2n1**2-1 +cos_theta**2)
    #try this:
    return (np.outer(np.ones_like(z)*zint, cos_theta) -
            np.outer(zint-z, np.sqrt(n2n1**2-1+cos_theta**2)))

def get_taus(cos_theta, n2n1=1./1.05):
    """
    Calculates the Fresnel reflectivity for s-polarized light incident on an
    interface with index ration n2n1.

    Inputs:
        -cos_theta: The _cosine_ of the angle of the incoming light. Float.
    Optional inputs:
        -n2n1: The ratio n2/n1 of the 2nd material's index n2 to the first's n1
    Returns:
        Float, same type (array or scalar) as cos_theta.
    """
    return 2./(1+np.sqrt(1+(n2n1**2-1)*cos_theta**-2))

def get_taup(cos_theta, n2n1=1./1.05):
    """
    Calculates the Fresnel reflectivity for p-polarized light incident on an
    interface with index ration n2n1.

    Inputs:
        -cos_theta: The _cosine_ of the angle of the incoming light. Float.
    Optional inputs:
        -n2n1: The ratio n2/n1 of the 2nd material's index n2 to the first's n1
    Returns:
        Float, same type (array or scalar) as cos_theta.
    """
    return 2*n2n1/(n2n1**2+np.sqrt(1-(1-n2n1**2)*cos_theta**-2))

def get_Kprefactor(z, cos_theta, zint=100.0, n2n1=0.95, get_hdet=False):
    """
    Internal function called by get_K; gets the prefactor in the integrand
    that is independent of which integral is being called.
    """

    phase = f_theta(cos_theta,zint,z,n2n1=n2n1)
    if get_hdet:
        to_return_rl = np.cos(phase)
        to_return_im = np.sin(phase)
    else:
        sqrt_costheta = np.outer(np.ones_like(z),np.sqrt(cos_theta))
        to_return_rl = np.cos(phase)*sqrt_costheta
        to_return_im = np.sin(phase)*sqrt_costheta
    return to_return_rl, to_return_im

def get_K(rho, z, alpha=1.0, zint=100.0, n2n1=0.95, get_hdet=False, K=1):
    """
    Internal function for calculating psf's. Returns various integrals that
    appear in Hell's psf calculation.
    Inputs:
        -rho: Rho in cylindrical coordinates. Float scalar or numpy.array.
        -z:   Z in cylindrical coordinates. Float scalar or numpy.array.

    Optional Inputs:
        -alpha: Float scalar on (0,pi/2). The acceptance angle of the lens.
        -zint: Float scalar on [0, inf). The distance of the len's
            unaberrated focal point from the interface.
        -n2n1: Float scalar on [0,inf) but really near 1. The ratio n2/n1
            of index mismatch between the sample (index n2) and the
            optical train (index n1).
        -get_hdet: Boolean. Set to True to get the detection portion of the
            psf; False to get the illumination portion of the psf.
        -K: 1, 2, or 3. Which of the 3 integrals to evaluate. Internal.
    Outputs:
        -integrand_rl: The integral's real      part; rho.shape numpy.array
        -integrand_im: The integral's imaginary part; . rho.shape numpy.array

    Comments:
        This is the only function that relies on rho,z being numpy.arrays,
        and it's just in a flag that I've added.... move to psf?
    """
    if type(rho) != np.ndarray or type(z) != np.ndarray or (rho.shape != z.shape):
        raise ValueError('rho and z must be np.arrays of same shape.')

    n1n2 = 1.0/n2n1

    rr = np.ravel(rho)
    zr = np.ravel(z)

    #Getting the array of points to quad at
    cos_theta = 0.5*(1-np.cos(alpha))*PTS+0.5*(1+np.cos(alpha))
    #[cosTheta,rho,z]

    Kprefactor_rl, Kprefactor_im = get_Kprefactor(z, cos_theta, zint=zint,
        n2n1=n2n1, get_hdet=get_hdet)

    if K==1:
        part_1 = j0(np.outer(rr,np.sqrt(1-cos_theta**2)))*\
            np.outer(np.ones_like(rr), 0.5*(get_taus(cos_theta,n2n1=n2n1)+\
            get_taup(cos_theta,n2n1=n2n1)*np.sqrt(1-n1n2**2*(1-cos_theta**2))))

        integrand_rl = Kprefactor_rl * part_1
        integrand_im = Kprefactor_im * part_1

    elif K==2:
        part_2=j2(np.outer(rr,np.sqrt(1-cos_theta**2)))*\
            np.outer(np.ones_like(rr),0.5*(get_taus(cos_theta,n2n1=n2n1)-\
            get_taup(cos_theta,n2n1=n2n1)*np.sqrt(1-n1n2**2*(1-cos_theta**2))))

        integrand_rl = Kprefactor_rl * part_2
        integrand_im = Kprefactor_im * part_2

    elif K==3:
        part_3=j1(np.outer(rho,np.sqrt(1-cos_theta**2)))*\
            np.outer(np.ones_like(rr), n1n2*get_taup(cos_theta,n2n1=n2n1)*\
            np.sqrt(1-cos_theta**2))

        integrand_rl = Kprefactor_rl * part_3
        integrand_im = Kprefactor_im * part_3

    else:
        raise RuntimeError('K=1,2,3 only...')

    big_wts = np.outer(np.ones_like(rr), WTS)
    kint_rl = (big_wts*integrand_rl).sum(axis=1) * 0.5*(1-np.cos(alpha))
    kint_im = (big_wts*integrand_im).sum(axis=1) * 0.5*(1-np.cos(alpha))

    return kint_rl.reshape(rho.shape), kint_im.reshape(rho.shape)

def get_hsym_asym(rho, z, get_hdet=False, **kwargs):
    """
    Gets the symmetric and asymmetric portions of the PSF. All distances
    (rho,z,zint) are in units of the 1/light wavevector.
    Inputs:
        -rho: Rho in cylindrical coordinates. Numpy.array.
        -z:   Z in cylindrical coordinates. Numpy.array.

    Optional Inputs:
        -alpha: Float scalar on (0,pi/2). The acceptance angle of the lens.
        -zint: Float scalar on [0, inf). The distance of the len's
            unaberrated focal point from the interface.
        -n2n1: Float scalar on [0,inf) but really near 1. The ratio n2/n1
            of index mismatch between the sample (index n2) and the
            optical train (index n1).
        -get_hdet: Boolean. Set to True to get the detection portion of the
            psf; False to get the illumination portion of the psf.

    Outputs:
        -hsym:  rho.shape numpy.array of the symmetric portion of the PSF
        -hasym: rho.shape numpy.array of the symmetric portion of the PSF
    """

    K1 = get_K(rho, z, K=1, get_hdet=get_hdet, **kwargs)
    K2 = get_K(rho, z, K=2, get_hdet=get_hdet, **kwargs)
    K3 = get_K(rho, z, K=3, get_hdet=get_hdet, **kwargs)

    hsym = K1[0]**2+K1[1]**2 + K2[0]**2+K2[1]**2 + 0.5*(K3[0]**2+K3[1]**2)
    hasym= 2*(K1[0]*K2[0] + K1[1]*K2[1]) + 0.5*(K3[0]**2+K3[1]**2)

    return hsym, hasym

def get_psf(x, y, z, kfki=0.89, zint=100.0, normalize=False, **kwargs):
    """
    Gets the PSF as calculated, for one set of points (x,y,z).
    Inputs:
        - x: Numpy array, the x-coordinate of the PSF in units of 1/ the
            wavevector of the incoming light.
        - y: Numpy array, the y-coordinate.
        - z: Numpy array, the z-coordinate.
        - kfki: Float scalar, the ratio of wavevectors of the outgoing light
            to the incoming light. Default is 0.89.
        - zint: Float scalar, the distance from the interface, in units of
            1/k_incoming. Default is 100.0
        - alpha: The opening angle of the lens.
        - n2n1: The ratio of the index in the 2nd medium to that in the first.
        - normalize: Boolean. Set to True to normalize the psf correctly,
            accounting for intensity variations with depth. This will give a
            psf that does not sum to 1.
    Outputs:
        - psf: x.shape numpy.array.

    Comments:
        (1) Note that the PSF is not necessarily centered on the z=0 pixel,
            since the calculation includes the shift.

        (2) If you want z-varying illumination of the psf then set
            normalize=True. This does the normalization by doing:
                hsym, hasym /= hsym.sum()
                hdet /= hdet.sum()
            and then calculating the psf that way. So if you want the
            intensity to be correct you need to use a large-ish array of
            roughly equally spaced points. Or do it manually by calling
            get_hsym_asym()
    """

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    hsym, hasym = get_hsym_asym(rho, z, zint=zint, get_hdet=False, **kwargs)
    hdet, toss  = get_hsym_asym(rho*kfki, z*kfki, zint=kfki*zint, get_hdet=True, **kwargs)

    if normalize:
        hasym /= hsym.sum()
        hsym /= hsym.sum()
        hdet /= hdet.sum()

    return (hsym + np.cos(2*phi)*hasym)*hdet

def get_psf_scalar(x, y, z, kfki=1., zint=100.0, normalize=False, **kwargs):
    """
    Gets an exact, wide-angle PSF for scalar (non-vectorial) light, i.e.
    ignoring the effects of polarization, for one set of points (x,y,z).
    This calculation also ignores the apodization factor for the ilm psf.
    Inputs:
        - x: Numpy array, the x-coordinate of the PSF in units of 1/ the
            wavevector of the incoming light.
        - y: Numpy array, the y-coordinate.
        - z: Numpy array, the z-coordinate.
        - kfki: Float scalar, the ratio of wavevectors of the outgoing light
            to the incoming light. Default is 1.0, which makes it 2x faster.
        - zint: Float scalar, the distance from the interface, in units of
            1/k_incoming. Default is 100.0
        - alpha: The opening angle of the lens.
        - n2n1: The ratio of the index in the 2nd medium to that in the first.
        - normalize: Boolean. Set to True to normalize the psf correctly,
            accounting for intensity variations with depth. This will give a
            psf that does not sum to 1.
    Outputs:
        - psf: x.shape numpy.array.

    Comments:
        (1) Note that the PSF is not necessarily centered on the z=0 pixel,
            since the calculation includes the shift.

        (2) If you want z-varying illumination of the psf then set
            normalize=True. This does the normalization by doing:
                hsym, hasym /= hsym.sum()
                hdet /= hdet.sum()
            and then calculating the psf that way. So if you want the
            intensity to be correct you need to use a large-ish array of
            roughly equally spaced points. Or do it manually by calling
            get_hsym_asym()
    """

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    K1 = get_K(rho, z, K=1,zint=zint,get_hdet=True, **kwargs)
    hilm = K1[0]*K1[0] + K1[1]*K1[1]

    if np.abs(kfki - 1.0) > 1e-13:
        Kdet = get_K(rho*kfki, z*kfki, K=1, zint=zint*kfki, get_hdet=True, **kwargs)
        hdet = Kdet[0]*Kdet[0] + Kdet[1]*Kdet[1]
    else:
        hdet = hilm.copy()

    if normalize:
        hilm /= hsym.sum()
        hdet /= hdet.sum()
        psf = hilm * hdet
    else:
        psf = hilm * hdet
        # psf /= psf.sum()

    return psf

def calculate_linescan_ilm_psf(y, z, pol_ang=0., scl=1.0, **kwargs):
    """
    I don't care what the psf is along the line direction since it's constant.
    So I can just return it along the orthogonal plane.
    But I need to maybe care if the polarization direction, line direction
    aren't the same.
    So we assume the polarization is in the
    """

    x_vals = PTS_HG*scl

    #I'm assuming that y,z are already some sort of meshgrid
    xg, yg, zg = [np.zeros( list(y.shape) + [x_vals.size] ) for a in xrange(3)]

    for a in xrange( x_vals.size ):
        xg[...,a] = x_vals[a]
        yg[...,a] = y.copy()
        zg[...,a] = z.copy()

    rho = np.sqrt(xg*xg + yg*yg)
    phi = np.arctan2(yg, xg)

    hsym, hasym = get_hsym_asym(rho, zg, get_hdet=False, **kwargs)

    hilm = hsym + np.cos(2*(phi-pol_ang))*hasym
    #multiply by weights... there is a better way to do this
    for a in xrange(x_vals.size):
        hilm[...,a] *= WTS_HG[a]

    return hilm.sum(axis=-1)*scl*2.

def calculate_linescan_psf(x, y, z, normalize=False, kfki=0.889, zint=100., **kwargs):
    """
    Make x,y,z  __1D__ numpy.arrays, with x the direction along the
    scan line. (to make the calculation faster since I dont' need the line
    ilm for each x).

    Inputs:
        - x, y, z: 1D numpy.arrays of the grid points to evaluate the PSF at.
            Floats, in units of 1/k_incoming. = lambda/2pi
    Optional Inputs:
        - normalize: Boolean. Set to True to include the effects of PSF
            normalization on the image intensity.
        - kfki: The ratio of the final light's wavevector to the incoming.
            Default is 0.889
        - zint: The position of the optical interface, in units of 1/k_incoming
    Other **kwargs
        - pol_ang: Float scalar of the polarization angle of the light with
            respect to the line direction (x). From calculate_linescan_ilm_psf;
            default is 0.
        - alpha: The aperture angle of the lens; default 1. From get_K().
            (in radians, 1.173 for our confocal / arcsin(n2n1))
        - n2n1: The index mismatch of the sample; default 0.95. From get_K().
            for our confocal, 1.4/1.518
    Outputs:
        - psf: 3D- numpy.array of the point-spread function. Indexing is
            psf[x,y,z].
    """

    #~~~Things I'd rather not have in this code.
    x3,y3,z3 = np.meshgrid(x, y, z, indexing='ij')
    y2,z2 = np.meshgrid(y, z, indexing='ij')

    rho3 = np.sqrt(x3*x3 + y3*y3)
    #~~~~

    hilm = calculate_linescan_ilm_psf(y2, z2, zint=zint, **kwargs)
    hdet, toss = get_hsym_asym(rho3*kfki, z3*kfki, zint=kfki*zint, get_hdet=True, **kwargs)

    if normalize:
        hilm /= hilm.sum()
        hdet /= hdet.sum()

    for a in xrange(x.size):
        hdet[a,...] *= hilm

    return hdet if normalize else hdet / hdet.sum()


class ExactLineScanConfocalPSF(PSF):
    def __init__(self, shape, kfki=0.889, zint=0., zscale=1.0, alpha=1.173,
            n2n1=1.4/1.518, polar_angle=0., *args, **kwargs):
        """

        PSF for line-scanning confocal microscopes that can be used with the
        cbamf framework.  Calculates the spatially varying point spread
        function for confocal microscopes and allows them to be applied to
        images as a convolution.

        Parameters:
        -----------
        shape : tuple
            Shape of the image in (z,y,x) pixel numbers (to be deprecated)

        kfki : float
            Ratio of outgoing to incoming light wavevectors, 2\pi/\lambda

        zint : float
            Pixel position of the optical interface where 0 is the edge of the
            image in the z direction

        zscale : float
            Scaling of the z pixel size, so that each pixel is located at
            zscale*(z - zint), such that the interface does not move with zscale

        alpha : float
            Aperture of the lens in radians, set by arcsin(n2n1)?

        n2n1 : float
            Ratio of the index mismatch across the optical interface. For typical
            glass with glycerol/water 80/20 mixture this is 1.4/1.518

        polar_angle : float
            the angle of the light polarization with respect to the scanning axis
        """
        params = np.array([kfki, zint, zscale, alpha, n2n1, polar_angle])
        super(ExactLineScanConfocalPSF, self).__init__(*args, params=params, shape=shape, **kwargs)

    def calculate_psf(self):
        pass
