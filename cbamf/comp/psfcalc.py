import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from scipy.special import j0,j1

from cbamf import const, util
from cbamf import interpolation
from cbamf.comp import psfs

def j2(x):
    """ A fast j2 defined in terms of other special functions """
    to_return = 2./(x+1e-15)*j1(x) - j0(x)
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

def f_theta(cos_theta, zint, z, n2n1=0.95):
    """
    """
    return (np.outer(np.ones_like(z)*zint, cos_theta) -
            np.outer(zint+z, csqrt(n2n1**2-1+cos_theta**2)))

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
    return 2./(1+csqrt(1+(n2n1**2-1)*cos_theta**-2))

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
    return 2*n2n1/(n2n1**2+csqrt(1-(1-n2n1**2)*cos_theta**-2))

def get_Kprefactor(z, cos_theta, zint=100.0, n2n1=0.95, get_hdet=False):
    """
    Internal function called by get_K; gets the prefactor in the integrand
    that is independent of which integral is being called.
    """

    phase = f_theta(cos_theta,zint,z,n2n1=n2n1)
    to_return = np.exp(-1j*phase)
    if not get_hdet:
        to_return *= np.outer(np.ones_like(z),np.sqrt(cos_theta))

    return to_return

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

    Kprefactor = get_Kprefactor(z, cos_theta, zint=zint, \
        n2n1=n2n1,get_hdet=get_hdet)

    if K==1:
        part_1 = j0(np.outer(rr,np.sqrt(1-cos_theta**2)))*\
            np.outer(np.ones_like(rr), 0.5*(get_taus(cos_theta,n2n1=n2n1)+\
            get_taup(cos_theta,n2n1=n2n1)*csqrt(1-n1n2**2*(1-cos_theta**2))))

        integrand = Kprefactor * part_1

    elif K==2:
        part_2=j2(np.outer(rr,np.sqrt(1-cos_theta**2)))*\
            np.outer(np.ones_like(rr),0.5*(get_taus(cos_theta,n2n1=n2n1)-\
            get_taup(cos_theta,n2n1=n2n1)*csqrt(1-n1n2**2*(1-cos_theta**2))))

        integrand = Kprefactor * part_2

    elif K==3:
        part_3=j1(np.outer(rho,np.sqrt(1-cos_theta**2)))*\
            np.outer(np.ones_like(rr), n1n2*get_taup(cos_theta,n2n1=n2n1)*\
            np.sqrt(1-cos_theta**2))

        integrand = Kprefactor * part_3

    else:
        raise ValueError('K=1,2,3 only...')

    big_wts=np.outer(np.ones_like(rr),WTS)
    kint = (big_wts*integrand).sum(axis=1) * 0.5*(1-np.cos(alpha))

    return kint.reshape(rho.shape)

def get_hsym_asym(rho, z, get_hdet=False, include_K3_det=True, **kwargs):
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
        -include_K3_det: Boolean. Flag to not calculate the `K3' component
            for the detection PSF, corresponding to (I think) a low-aperature
            focusing lens and no z-polarization of the focused light.

    Outputs:
        -hsym:  rho.shape numpy.array of the symmetric portion of the PSF
        -hasym: rho.shape numpy.array of the symmetric portion of the PSF
    """

    K1 = get_K(rho, z, K=1, get_hdet=get_hdet, **kwargs)
    K2 = get_K(rho, z, K=2, get_hdet=get_hdet, **kwargs)

    if get_hdet and not include_K3_det:
        K3 = 0*K1
    else:
        K3 = get_K(rho, z, K=3, get_hdet=get_hdet, **kwargs)

    hsym = K1*K1.conj() + K2*K2.conj() + 0.5*(K3*K3.conj())
    hasym= K1*K2.conj() + K2*K1.conj() + 0.5*(K3*K3.conj())

    return hsym.real, hasym.real #imaginary part should be 0

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
    hilm = np.real( K1*K1.conj() )

    if np.abs(kfki - 1.0) > 1e-13:
        Kdet = get_K(rho*kfki, z*kfki, K=1, zint=zint*kfki, get_hdet=True, **kwargs)
        hdet = np.real( Kdet*Kdet.conj() )
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

def calculate_linescan_ilm_psf(y,z, polar_angle=0., nlpts=1,
        pinhole_width=1, **kwargs):
    """
    Calculates the illumination PSF for a line-scanning confocal with the
    confocal line oriented along the x direction.
    Inputs:
        - y, z: Float numpy.arrays of the y- and z- points to evaluate
            the illumination PSF at, in units of 1/k. Arbitrary shape.
        - polar_angle: The angle of the illuminating light's polarization with
            respect to the x-axis (the line's orientation).
        - pinhole_width: The width of the geometric image of the line projected
            onto the sample, in units of 1/k. Default is 1. Set to 0 by setting
            nlpts = 1. The perfect line image is assumed to be a Gaussian.
        - nlpts: The number of points to use for Hermite-gauss quadrature over
            the line's width. Default is 1, corresponding to an infinitesmally
            thin line.
        - **kwargs: Paramters such as alpha, n2n1 that are passed to
            get_hsym_hasym
    Outputs:
        - hilm: Float numpy.array of the line illumination, of the same shape
            as y and z.
    """

    x_vals = PTS_HG

    #I'm assuming that y,z are already some sort of meshgrid
    xg, yg, zg = [np.zeros( list(y.shape) + [x_vals.size] ) for a in xrange(3)]
    hilm = np.zeros(xg.shape)

    for a in xrange(x_vals.size):
        xg[...,a] = x_vals[a]
        yg[...,a] = y.copy()
        zg[...,a] = z.copy()

    y_pinhole, wts_pinhole = np.polynomial.hermite.hermgauss(nlpts)
    y_pinhole *= np.sqrt(2)*pinhole_width
    wts_pinhole /= np.sqrt(np.pi)

    #Pinhole hermgauss first:
    for yp, wp in zip(y_pinhole, wts_pinhole):
        rho = np.sqrt(xg*xg + (yg-yp)*(yg-yp))
        phi = np.arctan2(yg,xg)

        hsym, hasym = get_hsym_asym(rho,zg,get_hdet = False, **kwargs)
        hilm += wp*(hsym + np.cos(2*(phi-polar_angle))*hasym)

    #Now line hermgauss
    for a in xrange(x_vals.size):
        hilm[...,a] *= WTS_HG[a]

    return hilm.sum(axis=-1)*2.

def calculate_linescan_psf(x, y, z, normalize=False, kfki=0.889, zint=100.,
        polar_angle=0., **kwargs):
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
        - polar_angle: Float scalar of the polarization angle of the light with
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

    hilm = calculate_linescan_ilm_psf(y2, z2, zint=zint, polar_angle=polar_angle, **kwargs)
    hdet, toss = get_hsym_asym(rho3*kfki, z3*kfki, zint=kfki*zint, get_hdet=True, **kwargs)

    if normalize:
        hilm /= hilm.sum()
        hdet /= hdet.sum()

    for a in xrange(x.size):
        hdet[a,...] *= hilm

    return hdet if normalize else hdet / hdet.sum()

def calculate_polychrome_linescan_psf(x, y, z, normalize=False, kfki=0.889,
        sigkf=0.1, zint=100., nkpts=3, **kwargs):
    """
    Calculates the full PSF for a line-scanning confocal with a polydisperse
    emission spectrum of the dye. The dye's emission spectrum is assumed to be
    Gaussian.
    Inputs:
        - x, y, z: 1D numpy.arrays of the grid points to evaluate the PSF at.
            Floats, in units of 1/k_incoming.
    Optional Inputs:
        - normalize: Boolean. Set to True to include the effects of PSF
            normalization on the image intensity.
        - kfki: The mean ratio of the final light's wavevector to the incoming.
            Default is 0.889
        - sigkf: Float scalar; sigma of kfki -- kfki values are kfki +- sigkf.
        - zint: The position of the optical interface, in units of 1/k_incoming
    Other **kwargs
        - polar_angle: Float scalar of the polarization angle of the light with
            respect to the line direction (x). From calculate_linescan_ilm_psf;
            default is 0.
        - alpha: The aperture angle of the lens; default 1. From get_K().
        - n2n1: The index mismatch of the sample; default 0.95. From get_K().
    Outputs:
        - psf: 3D- numpy.array of the point-spread function. Indexing is
            psf[x,y,z].
    """

    pts, wts = np.polynomial.hermite.hermgauss(nkpts)

    kfkipts = kfki + sigkf*np.sqrt(2)*pts
    wts /= np.sqrt(np.pi) #normalizing integral

    #~~~Things I'd rather not have in this code.
    x3,y3,z3 = np.meshgrid(x,y,z,indexing='ij')
    y2,z2 = np.meshgrid(y,z,indexing='ij')

    rho3 = np.sqrt(x3*x3+y3*y3)
    #~~~~

    hilm = calculate_linescan_ilm_psf(y2, z2, zint=zint, **kwargs)

    inner = [
        wts[a] * get_hsym_asym(
            rho3*kfkipts[a], z3*kfkipts[a], zint=kfkipts[a]*zint,
            get_hdet=True, **kwargs
        )[0]
        for a in xrange(nkpts)
    ]
    hdet = np.sum(inner, axis=0)

    if normalize:
        hilm /= hilm.sum()
        hdet /= hdet.sum()

    for a in xrange(x.size):
        hdet[a,...] *= hilm

    return hdet if normalize else hdet / hdet.sum()

def calculate_monochrome_brightfield_psf(x, y, z, normalize=False, **kwargs):
    """
    Calculate the PSF for brightfield, assuming illumination with unpolarized
    but vectorial light.
    Inputs:
        - x, y, z: numpy.arrays of the same shape at which to evaluate the psf
    Optional (**kwargs) arguments:
        - alpha
        - n2n1
        - zint

    Outputs:
        - psf: numpy.array of the psf, shape x.shape == y.shape == z.shape
    """

    rho = np.sqrt(x**2 + y**2)
    psf, toss = get_hsym_asym(rho, z, get_hdet=True, **kwargs)

    if normalize:
        norm = psf.sum(axis=(0,1)) #should be independent of z
        psf /= norm.max()

    return psf

def calculate_polychrome_brightfield_psf(x, y, z, k0=1., sigk=0.1, npts=3, **kwargs):
    pts,wts = np.polynomial.hermite.hermgauss(npts)

    kpts = k0 + sigk*np.sqrt(2)*pts
    wts /= np.sqrt(np.pi) #normalizing the integral

    inner = [
        wts[a] * calculate_monochrome_brightfield_psf(
            x*kpts[a], y*kpts[a],z*kpts[a], **kwargs
        ) for a in xrange(npts)
    ]
    psf = np.sum(inner, axis=0)

    return psf

def wrap_and_calc_psf(xpts, ypts, zpts, func, **kwargs):
    """
    Since all the PSFs have a cos2phi symmetry, which is also (loosely)
    the symmetry of a pixelated grid, ....
    Speeds up psf calculations by a factor of 4 for free / some broadcasting.
    Doesn't work for linescan psf because of the hdet bit...
    Inputs:
        - xpts: 1D N-element numpy.array of the x-points to
        - ypts:
        - zpts:
    """

    #1. Checking that everything is hunky-dory:
    for t in [xpts,ypts,zpts]:
        if len(t.shape) != 1:
            raise ValueError('xpts,ypts,zpts must be 1D.')

    for t in [xpts,ypts]:
        if t[0] != 0:
            raise ValueError('xpts[0],ypts[0] = 0 required.')

    xg,yg,zg = np.meshgrid(xpts,ypts,zpts, indexing='ij')
    xs, ys, zs = [ pts.size for pts in [xpts,ypts,zpts] ]
    to_return = np.zeros([2*xs-1, 2*ys-1, zs])

    #2. Calculate:
    up_corner_psf = func(xg,yg,zg, **kwargs)

    to_return[xs-1:,ys-1:,:] = up_corner_psf.copy()                 #x>0, y>0
    to_return[:xs-1,ys-1:,:] = up_corner_psf[-1:0:-1,:,:].copy()    #x<0, y>0
    to_return[xs-1:,:ys-1,:] = up_corner_psf[:,-1:0:-1,:].copy()    #x>0, y<0
    to_return[:xs-1,:ys-1,:] = up_corner_psf[-1:0:-1,-1:0:-1,:].copy()#x<0,y<0

    return to_return

