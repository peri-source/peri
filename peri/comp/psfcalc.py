from builtins import range

import warnings
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from scipy.special import j0,j1, la_roots

from peri import util
from peri import interpolation
from peri.comp import psfs

def j2(x):
    """ A fast j2 defined in terms of other special functions """
    to_return = 2./(x+1e-15)*j1(x) - j0(x)
    to_return[x==0] = 0
    return to_return

#Two methods for calculating quadrature points for integration over the
#illuminating line:
def calc_pts_hg(npts=20):
    """Returns Hermite-Gauss quadrature points for even functions"""
    pts_hg, wts_hg = np.polynomial.hermite.hermgauss(npts*2)
    pts_hg = pts_hg[npts:]
    wts_hg = wts_hg[npts:] * np.exp(pts_hg*pts_hg)
    return pts_hg, wts_hg

def calc_pts_lag(npts=20):
    """
    Returns Gauss-Laguerre quadrature points rescaled for line scan integration

    Parameters
    ----------
        npts : {15, 20, 25}, optional
            The number of points to

    Notes
    -----
        The scale is set internally as the best rescaling for a line scan
        integral; it was checked numerically for the allowed npts.
        Acceptable pts/scls/approximate line integral scan error:
        (pts,   scl  )      :         ERR
        ------------------------------------
        (15, 0.072144)      :       0.002193
        (20, 0.051532)      :       0.001498
        (25, 0.043266)      :       0.001209

        The previous HG(20) error was ~0.13ish
    """
    scl = { 15:0.072144,
            20:0.051532,
            25:0.043266}[npts]
    pts0, wts0 = np.polynomial.laguerre.laggauss(npts)
    pts = np.sinh(pts0*scl)
    wts = scl*wts0*np.cosh(pts0*scl)*np.exp(pts0)
    return pts, wts

#######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#######
#                        Electric Field Focus Integrals
#######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#######

def f_theta(cos_theta, zint, z, n2n1=0.95, sph6_ab=None, **kwargs):
    """
    Returns the wavefront aberration for an aberrated, defocused lens.

    Calculates the portions of the wavefront distortion due to z, theta
    only, for a lens with defocus and spherical aberration induced by
    coverslip mismatch. (The rho portion can be analytically integrated
    to Bessels.)

    Parameters
    ----------
        cos_theta : numpy.ndarray.
            The N values of cos(theta) at which to compute f_theta.
        zint : Float
            The position of the lens relative to the interface.
        z : numpy.ndarray
            The M z-values to compute f_theta at. `z.size` is unrelated
            to `cos_theta.size`
        n2n1: Float, optional
            The ratio of the index of the immersed medium to the optics.
            Default is 0.95
        sph6_ab : Float or None, optional
            Set sph6_ab to a nonzero value to add residual 6th-order
            spherical aberration that is proportional to sph6_ab. Default
            is None (i.e. doesn't calculate).

    Returns
    -------
        wvfront : numpy.ndarray
            The aberrated wavefront, as a function of theta and z.
            Shape is [z.size, cos_theta.size]
    """
    wvfront = (np.outer(np.ones_like(z)*zint, cos_theta) -
            np.outer(zint+z, csqrt(n2n1**2-1+cos_theta**2)))
    if (sph6_ab is not None) and (not np.isnan(sph6_ab)):
        sec2_theta = 1.0/(cos_theta*cos_theta)
        wvfront += sph6_ab * (sec2_theta-1)*(sec2_theta-2)*cos_theta
    #Ensuring evanescent waves are always suppressed:
    if wvfront.dtype == np.dtype('complex128'):
        wvfront.imag = -np.abs(wvfront.imag)
    return wvfront

def get_taus(cos_theta, n2n1=0.95):
    """
    Calculates the Fresnel reflectivity for s-polarized light incident on an
    interface with index ration n2n1.

    Parameters
    ----------
        cos_theta : Float or numpy.ndarray
            The _cosine_ of the angle of the incoming light. Float.
        n2n1 : Float, optional
            The ratio n2/n1 of the 2nd material's index n2 to the first's n1
            Default is 0.95

    Returns
    -------
        Float or numpy.ndarray
            The reflectivity, in the same type (ndarray or Float) and
            shape as cos_theta
    """
    return 2./(1+csqrt(1+(n2n1**2-1)*cos_theta**-2))

def get_taup(cos_theta, n2n1=0.95):
    """
    Calculates the Fresnel reflectivity for p-polarized light incident on an
    interface with index ration n2n1.

    Parameters
    ----------
        cos_theta : Float or numpy.ndarray
            The _cosine_ of the angle of the incoming light. Float.
        n2n1 : Float, optional
            The ratio n2/n1 of the 2nd material's index n2 to the first's n1
            Default is 0.95

    Returns
    -------
        Float or numpy.ndarray
            The reflectivity, in the same type (ndarray or Float) and
            shape as cos_theta
    """
    return 2*n2n1/(n2n1**2+csqrt(1-(1-n2n1**2)*cos_theta**-2))

def get_Kprefactor(z, cos_theta, zint=100.0, n2n1=0.95, get_hdet=False,
        **kwargs):
    """
    Returns a prefactor in the electric field integral.

    This is an internal function called by get_K. The returned prefactor
    in the integrand is independent of which integral is being called;
    it is a combination of the exp(1j*phase) and apodization.

    Parameters
    ----------
        z : numpy.ndarray
            The values of z (distance along optical axis) at which to
            calculate the prefactor. Size is unrelated to the size of
            `cos_theta`
        cos_theta : numpy.ndarray
            The values of cos(theta) (i.e. position on the incoming
            focal spherical wavefront) at which to calculate the
            prefactor. Size is unrelated to the size of `z`
        zint : Float, optional
            The position of the optical interface, in units of 1/k.
            Default is 100.
        n2n1 : Float, optional
            The ratio of the index mismatch between the optics (n1) and
            the sample (n2). Default is 0.95
        get_hdet : Bool, optional
            Set to True to calculate the detection prefactor vs the
            illumination prefactor (i.e. False to include apodization).
            Default is False

    Returns
    -------
        numpy.ndarray
            The prefactor, of size [`z.size`, `cos_theta.size`], sampled
            at the values [`z`, `cos_theta`]
    """
    phase = f_theta(cos_theta, zint, z, n2n1=n2n1, **kwargs)
    to_return = np.exp(-1j*phase)
    if not get_hdet:
        to_return *= np.outer(np.ones_like(z),np.sqrt(cos_theta))

    return to_return

def get_K(rho, z, alpha=1.0, zint=100.0, n2n1=0.95, get_hdet=False, K=1,
        Kprefactor=None, return_Kprefactor=False, npts=20, **kwargs):
    """
    Calculates one of three electric field integrals.

    Internal function for calculating point spread functions. Returns
    one of three electric field integrals that describe the electric
    field near the focus of a lens; these integrals appear in Hell's psf
    calculation.

    Parameters
    ----------
        rho : numpy.ndarray
            Rho in cylindrical coordinates, in units of 1/k.
        z : numpy.ndarray
            Z in cylindrical coordinates, in units of 1/k. `rho` and
            `z` must be the same shape

        alpha : Float, optional
            The acceptance angle of the lens, on (0,pi/2). Default is 1.
        zint : Float, optional
            The distance of the len's unaberrated focal point from the
            optical interface, in units of 1/k. Default is 100.
        n2n1 : Float, optional
            The ratio n2/n1 of the index mismatch between the sample
            (index n2) and the optical train (index n1). Must be on
            [0,inf) but should be near 1. Default is 0.95
        get_hdet : Bool, optional
            Set to True to get the detection portion of the psf; False
            to get the illumination portion of the psf. Default is True
        K : {1, 2, 3}, optional
            Which of the 3 integrals to evaluate. Default is 1
        Kprefactor : numpy.ndarray or None
            This array is calculated internally and optionally returned;
            pass it back to avoid recalculation and increase speed. Default
            is None, i.e. calculate it internally.
        return_Kprefactor : Bool, optional
            Set to True to also return the Kprefactor (parameter above)
            to speed up the calculation for the next values of K. Default
            is False
        npts : Int, optional
            The number of points to use for Gauss-Legendre quadrature of
            the integral. Default is 20, which is a good number for x,y,z
            less than 100 or so.

    Returns
    -------
        kint : numpy.ndarray
            The integral K_i; rho.shape numpy.array
        [, Kprefactor] : numpy.ndarray
            The prefactor that is independent of which integral is being
            calculated but does depend on the parameters; can be passed
            back to the function for speed.

    Notes
    -----
        npts=20 gives double precision (no difference between 20, 30, and
        doing all the integrals with scipy.quad). The integrals are only
        over the acceptance angle of the lens, so for moderate x,y,z they
        don't vary too rapidly. For x,y,z, zint large compared to 100, a
        higher npts might be necessary.
    """
    # Comments:
        # This is the only function that relies on rho,z being numpy.arrays,
        # and it's just in a flag that I've added.... move to psf?
    if type(rho) != np.ndarray or type(z) != np.ndarray or (rho.shape != z.shape):
        raise ValueError('rho and z must be np.arrays of same shape.')

    pts, wts = np.polynomial.legendre.leggauss(npts)
    n1n2 = 1.0/n2n1

    rr = np.ravel(rho)
    zr = np.ravel(z)

    #Getting the array of points to quad at
    cos_theta = 0.5*(1-np.cos(alpha))*pts+0.5*(1+np.cos(alpha))
    #[cos_theta,rho,z]

    if Kprefactor is None:
        Kprefactor = get_Kprefactor(z, cos_theta, zint=zint, \
            n2n1=n2n1,get_hdet=get_hdet, **kwargs)

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

    big_wts=np.outer(np.ones_like(rr), wts)
    kint = (big_wts*integrand).sum(axis=1) * 0.5*(1-np.cos(alpha))

    if return_Kprefactor:
        return kint.reshape(rho.shape), Kprefactor
    else:
        return kint.reshape(rho.shape)

#######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#######
#                          Confocal PSF Calculations
#######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#######

def get_hsym_asym(rho, z, get_hdet=False, include_K3_det=True, **kwargs):
    """
    Calculates the symmetric and asymmetric portions of a confocal PSF.

    Parameters
    ----------
        rho : numpy.ndarray
            Rho in cylindrical coordinates, in units of 1/k.
        z : numpy.ndarray
            Z in cylindrical coordinates, in units of 1/k. Must be the
            same shape as `rho`
        get_hdet : Bool, optional
            Set to True to get the detection portion of the psf; False
            to get the illumination portion of the psf. Default is True
        include_K3_det : Bool, optional.
            Flag to not calculate the `K3' component for the detection
            PSF, corresponding to (I think) a low-aperature focusing
            lens and no z-polarization of the focused light. Default
            is True, i.e. calculates the K3 component as if the focusing
            lens is high-aperture

    Other Parameters
    ----------------
        alpha : Float, optional
            The acceptance angle of the lens, on (0,pi/2). Default is 1.
        zint : Float, optional
            The distance of the len's unaberrated focal point from the
            optical interface, in units of 1/k. Default is 100.
        n2n1 : Float, optional
            The ratio n2/n1 of the index mismatch between the sample
            (index n2) and the optical train (index n1). Must be on
            [0,inf) but should be near 1. Default is 0.95

    Returns
    -------
        hsym : numpy.ndarray
            `rho`.shape numpy.array of the symmetric portion of the PSF
        hasym : numpy.ndarray
            `rho`.shape numpy.array of the asymmetric portion of the PSF
    """

    K1, Kprefactor = get_K(rho, z, K=1, get_hdet=get_hdet, Kprefactor=None,
            return_Kprefactor=True, **kwargs)
    K2 = get_K(rho, z, K=2, get_hdet=get_hdet, Kprefactor=Kprefactor,
            return_Kprefactor=False, **kwargs)

    if get_hdet and not include_K3_det:
        K3 = 0*K1
    else:
        K3 = get_K(rho, z, K=3, get_hdet=get_hdet, Kprefactor=Kprefactor,
            return_Kprefactor=False, **kwargs)

    hsym = K1*K1.conj() + K2*K2.conj() + 0.5*(K3*K3.conj())
    hasym= K1*K2.conj() + K2*K1.conj() + 0.5*(K3*K3.conj())

    return hsym.real, hasym.real #imaginary part should be 0

def calculate_pinhole_psf(x, y, z, kfki=0.89, zint=100.0, normalize=False,
        **kwargs):
    """
    Calculates the perfect-pinhole PSF, for a set of points (x,y,z).

    Parameters
    -----------
        x : numpy.ndarray
            The x-coordinate of the PSF in units of 1/ the wavevector of
            the incoming light.
        y : numpy.ndarray
            The y-coordinate.
        z : numpy.ndarray
            The z-coordinate.
        kfki : Float
            The (scalar) ratio of wavevectors of the outgoing light to the
            incoming light. Default is 0.89.
        zint : Float
            The (scalar) distance from the interface, in units of
            1/k_incoming. Default is 100.0
        normalize : Bool
            Set to True to normalize the psf correctly, accounting for
            intensity variations with depth. This will give a psf that does
            not sum to 1.

    Other Parameters
    ----------------
        alpha : Float
            The opening angle of the lens. Default is 1.
        n2n1 : Float
            The ratio of the index in the 2nd medium to that in the first.
            Default is 0.95

    Returns
    -------
        psf : numpy.ndarray, of shape x.shape

    Comments
    --------
        (1) The PSF is not necessarily centered on the z=0 pixel, since the
            calculation includes the shift.

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

def get_polydisp_pts_wts(kfki, sigkf, dist_type='gaussian', nkpts=3):
    """
    Calculates a set of Gauss quadrature points & weights for polydisperse
    light.

    Returns a list of points and weights of the final wavevector's distri-
    bution, in units of the initial wavevector.

    Parameters
    ----------
        kfki : Float
            The mean of the polydisperse outgoing wavevectors.
        sigkf : Float
            The standard dev. of the polydisperse outgoing wavevectors.
        dist_type : {`gaussian`, `gamma`}, optional
            The distribution, gaussian or gamma, of the wavevectors.
            Default is `gaussian`
        nkpts : Int, optional
            The number of quadrature points to use. Default is 3
    Returns
    -------
        kfkipts : numpy.ndarray
            The Gauss quadrature points at which to calculate kfki.
        wts : numpy.ndarray
            The associated Gauss quadrature weights.
    """
    if dist_type.lower() == 'gaussian':
        pts, wts = np.polynomial.hermite.hermgauss(nkpts)
        kfkipts = np.abs(kfki + sigkf*np.sqrt(2)*pts)
    elif dist_type.lower() == 'laguerre' or dist_type.lower() == 'gamma':
        k_scale = sigkf**2/kfki
        associated_order = kfki**2/sigkf**2 - 1
        #Associated Laguerre with alpha >~170 becomes numerically unstable, so:
        max_order=150
        if associated_order > max_order or associated_order < (-1+1e-3):
            warnings.warn('Numerically unstable sigk, clipping', RuntimeWarning)
            associated_order = np.clip(associated_order, -1+1e-3, max_order)
        kfkipts, wts = la_roots(nkpts, associated_order)
        kfkipts *= k_scale
    else:
        raise ValueError('dist_type must be either gaussian or laguerre')
    return kfkipts, wts/wts.sum()

def calculate_polychrome_pinhole_psf(x, y, z, normalize=False, kfki=0.889,
        sigkf=0.1, zint=100., nkpts=3, dist_type='gaussian', **kwargs):
    """
    Calculates the perfect-pinhole PSF, for a set of points (x,y,z).

    Parameters
    -----------
        x : numpy.ndarray
            The x-coordinate of the PSF in units of 1/ the wavevector of
            the incoming light.
        y : numpy.ndarray
            The y-coordinate.
        z : numpy.ndarray
            The z-coordinate.
        kfki : Float
            The mean ratio of the outgoing light's wavevector to the incoming
            light's. Default is 0.89.
        sigkf : Float
            Standard deviation of kfki; the distribution of the light values
            will be approximately kfki +- sigkf.
        zint : Float
            The (scalar) distance from the interface, in units of
            1/k_incoming. Default is 100.0
        dist_type: The distribution type of the polychromatic light.
            Can be one of 'laguerre'/'gamma' or 'gaussian.' If 'gaussian'
            the resulting k-values are taken in absolute value. Default
            is 'gaussian.'
        normalize : Bool
            Set to True to normalize the psf correctly, accounting for
            intensity variations with depth. This will give a psf that does
            not sum to 1. Default is False.

    Other Parameters
    ----------------
        alpha : Float
            The opening angle of the lens. Default is 1.
        n2n1 : Float
            The ratio of the index in the 2nd medium to that in the first.
            Default is 0.95

    Returns
    -------
        psf : numpy.ndarray, of shape x.shape

    Comments
    --------
        (1) The PSF is not necessarily centered on the z=0 pixel, since the
            calculation includes the shift.

        (2) If you want z-varying illumination of the psf then set
            normalize=True. This does the normalization by doing:
                hsym, hasym /= hsym.sum()
                hdet /= hdet.sum()
            and then calculating the psf that way. So if you want the
            intensity to be correct you need to use a large-ish array of
            roughly equally spaced points. Or do it manually by calling
            get_hsym_asym()
    """
    #0. Setup
    kfkipts, wts = get_polydisp_pts_wts(kfki, sigkf, dist_type=dist_type,
            nkpts=nkpts)
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    #1. Hilm
    hsym, hasym = get_hsym_asym(rho, z, zint=zint, get_hdet=False, **kwargs)
    hilm = (hsym + np.cos(2*phi)*hasym)

    #2. Hdet
    hdet_func = lambda kfki: get_hsym_asym(rho*kfki, z*kfki,
                zint=kfki*zint, get_hdet=True, **kwargs)[0]
    inner = [wts[a] * hdet_func(kfkipts[a]) for a in range(nkpts)]
    hdet = np.sum(inner, axis=0)

    #3. Normalize and return
    if normalize:
        hilm /= hilm.sum()
        hdet /= hdet.sum()
    psf = hdet * hilm
    return psf if normalize else psf / psf.sum()

def get_psf_scalar(x, y, z, kfki=1., zint=100.0, normalize=False, **kwargs):
    """
    Calculates a scalar (non-vectorial light) approximation to a confocal PSF

    The calculation is approximate, since it ignores the effects of
    polarization and apodization, but should be ~3x faster.

    Parameters
    ----------
        x : numpy.ndarray
            The x-coordinate of the PSF in units of 1/ the wavevector
            of the incoming light.
        y : numpy.ndarray
            The y-coordinate of the PSF in units of 1/ the wavevector
            of the incoming light. Must be the same shape as `x`.
        z : numpy.ndarray
            The z-coordinate of the PSF in units of 1/ the wavevector
            of the incoming light. Must be the same shape as `x`.
        kfki : Float, optional
            The ratio of wavevectors of the outgoing light to the
            incoming light. Set to 1.0 to speed up the calculation
            by another factor of 2. Default is 1.0
        zint : Float, optional
            The distance from to the optical interface, in units of
            1/k_incoming. Default is 100.
        normalize : Bool
            Set to True to normalize the psf correctly, accounting for
            intensity variations with depth. This will give a psf that does
            not sum to 1. Default is False.
        alpha : Float
            The opening angle of the lens. Default is 1.
        n2n1 : Float
            The ratio of the index in the 2nd medium to that in the first.
            Default is 0.95

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
        Kdet = get_K(rho*kfki, z*kfki, K=1, zint=zint*kfki, get_hdet=True,
                **kwargs)
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
        pinhole_width=1, use_laggauss=False, **kwargs):
    """
    Calculates the illumination PSF for a line-scanning confocal with the
    confocal line oriented along the x direction.

    Parameters
    ----------
        y : numpy.ndarray
            The y points (in-plane, perpendicular to the line direction)
            at which to evaluate the illumination PSF, in units of 1/k.
            Arbitrary shape.
        z : numpy.ndarray
            The z points (optical axis) at which to evaluate the illum-
            ination PSF, in units of 1/k. Must be the same shape as `y`
        polar_angle : Float, optional
            The angle of the illuminating light's polarization with
            respect to the line's orientation along x. Default is 0.
        pinhole_width : Float, optional
            The width of the geometric image of the line projected onto
            the sample, in units of 1/k. Default is 1. The perfect line
            image is assumed to be a Gaussian. If `nlpts` is set to 1,
            the line will always be of zero width.
        nlpts : Int, optional
            The number of points to use for Hermite-gauss quadrature over
            the line's width. Default is 1, corresponding to a zero-width
            line.
        use_laggauss : Bool, optional
            Set to True to use a more-accurate sinh'd Laguerre-Gauss
            quadrature for integration over the line's length (more accurate
            in the same amount of time). Default is False for backwards
            compatibility.  FIXME what did we do here?

    Other Parameters
    ----------------
        alpha : Float, optional
            The acceptance angle of the lens, on (0,pi/2). Default is 1.
        zint : Float, optional
            The distance of the len's unaberrated focal point from the
            optical interface, in units of 1/k. Default is 100.
        n2n1 : Float, optional
            The ratio n2/n1 of the index mismatch between the sample
            (index n2) and the optical train (index n1). Must be on
            [0,inf) but should be near 1. Default is 0.95

    Returns
    -------
        hilm : numpy.ndarray
            The line illumination, of the same shape as y and z.
    """
    if use_laggauss:
        x_vals, wts = calc_pts_lag()
    else:
        x_vals, wts = calc_pts_hg()

    #I'm assuming that y,z are already some sort of meshgrid
    xg, yg, zg = [np.zeros( list(y.shape) + [x_vals.size] ) for a in range(3)]
    hilm = np.zeros(xg.shape)

    for a in range(x_vals.size):
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
    for a in range(x_vals.size):
        hilm[...,a] *= wts[a]

    return hilm.sum(axis=-1)*2.

def calculate_linescan_psf(x, y, z, normalize=False, kfki=0.889, zint=100.,
        polar_angle=0., wrap=True, **kwargs):
    """
    Calculates the point spread function of a line-scanning confocal.

    Make x,y,z  __1D__ numpy.arrays, with x the direction along the
    scan line. (to make the calculation faster since I dont' need the line
    ilm for each x).

    Parameters
    ----------
        x : numpy.ndarray
            _One_dimensional_ array of the x grid points (along the line
            illumination) at which to evaluate the psf. In units of
            1/k_incoming.
        y : numpy.ndarray
            _One_dimensional_ array of the y grid points (in plane,
            perpendicular to the line illumination) at which to evaluate
            the psf. In units of 1/k_incoming.
        z : numpy.ndarray
            _One_dimensional_ array of the z grid points (along the
            optical axis) at which to evaluate the psf. In units of
            1/k_incoming.
        normalize : Bool, optional
            Set to True to include the effects of PSF normalization on
            the image intensity. Default is False.
        kfki : Float, optional
            The ratio of the final light's wavevector to the incoming.
            Default is 0.889
        zint : Float, optional
            The position of the optical interface, in units of 1/k_incoming
            Default is 100.
        wrap : Bool, optional
            If True, wraps the psf calculation for speed, assuming that
            the input x, y are regularly-spaced points. If x,y are not
            regularly spaced then `wrap` must be set to False. Default is True.
        polar_angle : Float, optional
            The polarization angle of the light (radians) with respect to
            the line direction (x). Default is 0.

    Other Parameters
    ----------------
        alpha : Float
            The opening angle of the lens. Default is 1.
        n2n1 : Float
            The ratio of the index in the 2nd medium to that in the first.
            Default is 0.95

    Returns
    -------
        numpy.ndarray
            A 3D- numpy.array of the point-spread function. Indexing is
            psf[x,y,z]; shape is [x.size, y,size, z.size]
    """

    #0. Set up vecs
    if wrap:
        xpts = vec_to_halfvec(x)
        ypts = vec_to_halfvec(y)
        x3, y3, z3 = np.meshgrid(xpts, ypts, z, indexing='ij')
    else:
        x3,y3,z3 = np.meshgrid(x, y, z, indexing='ij')
    rho3 = np.sqrt(x3*x3 + y3*y3)

    #1. Hilm
    if wrap:
        y2,z2 = np.meshgrid(ypts, z, indexing='ij')
        hilm0 = calculate_linescan_ilm_psf(y2, z2, zint=zint,
                polar_angle=polar_angle, **kwargs)
        if ypts[0] == 0:
            hilm = np.append(hilm0[-1:0:-1], hilm0, axis=0)
        else:
            hilm = np.append(hilm0[::-1], hilm0, axis=0)
    else:
        y2,z2 = np.meshgrid(y, z, indexing='ij')
        hilm = calculate_linescan_ilm_psf(y2, z2, zint=zint,
                polar_angle=polar_angle, **kwargs)

    #2. Hdet
    if wrap:
        #Lambda function that ignores its args but still returns correct values
        func = lambda *args: get_hsym_asym(rho3*kfki, z3*kfki, zint=kfki*zint,
                    get_hdet=True, **kwargs)[0]
        hdet = wrap_and_calc_psf(xpts, ypts, z, func)
    else:
        hdet, toss = get_hsym_asym(rho3*kfki, z3*kfki, zint=kfki*zint,
                get_hdet=True, **kwargs)

    if normalize:
        hilm /= hilm.sum()
        hdet /= hdet.sum()

    for a in range(x.size):
        hdet[a] *= hilm

    return hdet if normalize else hdet / hdet.sum()

def calculate_polychrome_linescan_psf(x, y, z, normalize=False, kfki=0.889,
        sigkf=0.1, zint=100., nkpts=3, dist_type='gaussian', wrap=True,
        **kwargs):
    """
    Calculates the point spread function of a line-scanning confocal with
    polydisperse dye emission.

    Make x,y,z  __1D__ numpy.arrays, with x the direction along the
    scan line. (to make the calculation faster since I dont' need the line
    ilm for each x).

    Parameters
    ----------
        x : numpy.ndarray
            _One_dimensional_ array of the x grid points (along the line
            illumination) at which to evaluate the psf. In units of
            1/k_incoming.
        y : numpy.ndarray
            _One_dimensional_ array of the y grid points (in plane,
            perpendicular to the line illumination) at which to evaluate
            the psf. In units of 1/k_incoming.
        z : numpy.ndarray
            _One_dimensional_ array of the z grid points (along the
            optical axis) at which to evaluate the psf. In units of
            1/k_incoming.
        normalize : Bool, optional
            Set to True to include the effects of PSF normalization on
            the image intensity. Default is False.
        kfki : Float, optional
            The mean of the ratio of the final light's wavevector to the
            incoming. Default is 0.889
        sigkf : Float, optional
            The standard deviation of the ratio of the final light's
            wavevector to the incoming. Default is 0.1
        zint : Float, optional
            The position of the optical interface, in units of 1/k_incoming
            Default is 100.
        dist_type : {`gaussian`, `gamma`}, optional
            The distribution of the outgoing light. If 'gaussian' the
            resulting k-values are taken in absolute value. Default
            is `gaussian`
        wrap : Bool, optional
            If True, wraps the psf calculation for speed, assuming that
            the input x, y are regularly-spaced points. If x,y are not
            regularly spaced then `wrap` must be set to False. Default is True.

    Other Parameters
    ----------------
        polar_angle : Float, optional
            The polarization angle of the light (radians) with respect to
            the line direction (x). Default is 0.
        alpha : Float
            The opening angle of the lens. Default is 1.
        n2n1 : Float
            The ratio of the index in the 2nd medium to that in the first.
            Default is 0.95

    Returns
    -------
        numpy.ndarray
            A 3D- numpy.array of the point-spread function. Indexing is
            psf[x,y,z]; shape is [x.size, y,size, z.size]
    Notes
    -----
        Neither distribution type is perfect. If sigkf/k0 is big (>0.5ish)
        then part of the Gaussian is negative. To avoid issues an abs() is
        taken, but then the actual mean and variance are not what is
        supplied. Conversely, if sigkf/k0 is small (<0.0815), then the
        requisite associated Laguerre quadrature becomes unstable. To
        prevent this sigkf/k0 is effectively clipped to be > 0.0815.
    """
    kfkipts, wts = get_polydisp_pts_wts(kfki, sigkf, dist_type=dist_type,
            nkpts=nkpts)

    #0. Set up vecs
    if wrap:
        xpts = vec_to_halfvec(x)
        ypts = vec_to_halfvec(y)
        x3, y3, z3 = np.meshgrid(xpts, ypts, z, indexing='ij')
    else:
        x3,y3,z3 = np.meshgrid(x, y, z, indexing='ij')
    rho3 = np.sqrt(x3*x3 + y3*y3)

    #1. Hilm
    if wrap:
        y2,z2 = np.meshgrid(ypts, z, indexing='ij')
        hilm0 = calculate_linescan_ilm_psf(y2, z2, zint=zint, **kwargs)
        if ypts[0] == 0:
            hilm = np.append(hilm0[-1:0:-1], hilm0, axis=0)
        else:
            hilm = np.append(hilm0[::-1], hilm0, axis=0)
    else:
        y2,z2 = np.meshgrid(y, z, indexing='ij')
        hilm = calculate_linescan_ilm_psf(y2, z2, zint=zint, **kwargs)

    #2. Hdet
    if wrap:
        #Lambda function that ignores its args but still returns correct values
        func = lambda x,y,z, kfki=1.: get_hsym_asym(rho3*kfki, z3*kfki,
                zint=kfki*zint, get_hdet=True, **kwargs)[0]
        hdet_func = lambda kfki: wrap_and_calc_psf(xpts,ypts,z, func, kfki=kfki)
    else:
        hdet_func = lambda kfki: get_hsym_asym(rho3*kfki, z3*kfki,
                zint=kfki*zint, get_hdet=True, **kwargs)[0]
    #####
    inner = [wts[a] * hdet_func(kfkipts[a]) for a in range(nkpts)]
    hdet = np.sum(inner, axis=0)

    if normalize:
        hilm /= hilm.sum()
        hdet /= hdet.sum()
    for a in range(x.size):
        hdet[a] *= hilm

    return hdet if normalize else hdet / hdet.sum()


#######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#######
#                              Utility Functions
#######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#######

def wrap_and_calc_psf(xpts, ypts, zpts, func, **kwargs):
    """
    Wraps a point-spread function in x and y.

    Speeds up psf calculations by a factor of 4 for free / some broadcasting
    by exploiting the x->-x, y->-y symmetry of a psf function. Pass x and y
    as the positive (say) values of the coordinates at which to evaluate func,
    and it will return the function sampled at [x[::-1]] + x. Note it is not
    wrapped in z.

    Parameters
    ----------
        xpts : numpy.ndarray
            1D N-element numpy.array of the x-points to evaluate func at.
        ypts : numpy.ndarray
            y-points to evaluate func at.
        zpts : numpy.ndarray
            z-points to evaluate func at.
        func : function
            The function to evaluate and wrap around. Syntax must be
            func(x,y,z, **kwargs)
        **kwargs : Any parameters passed to the function.

    Outputs
    -------
        to_return : numpy.ndarray
            The wrapped and calculated psf, of shape
            [2*x.size - x0, 2*y.size - y0, z.size], where x0=1 if x[0]=0, etc.

    Notes
    -----
    The coordinates should be something like numpy.arange(start, stop, diff),
    with start near 0. If x[0]==0, all of x is calcualted but only x[1:]
    is wrapped (i.e. it works whether or not x[0]=0).

    This doesn't work directly for a linescan psf because the illumination
    portion is not like a grid. However, the illumination and detection
    are already combined with wrap_and_calc in calculate_linescan_psf etc.
    """
    #1. Checking that everything is hunky-dory:
    for t in [xpts,ypts,zpts]:
        if len(t.shape) != 1:
            raise ValueError('xpts,ypts,zpts must be 1D.')

    dx = 1 if xpts[0]==0 else 0
    dy = 1 if ypts[0]==0 else 0

    xg,yg,zg = np.meshgrid(xpts,ypts,zpts, indexing='ij')
    xs, ys, zs = [ pts.size for pts in [xpts,ypts,zpts] ]
    to_return = np.zeros([2*xs-dx, 2*ys-dy, zs])

    #2. Calculate:
    up_corner_psf = func(xg,yg,zg, **kwargs)

    to_return[xs-dx:,ys-dy:,:] = up_corner_psf.copy()                     #x>0, y>0
    if dx == 0:
        to_return[:xs-dx,ys-dy:,:] = up_corner_psf[::-1,:,:].copy()       #x<0, y>0
    else:
        to_return[:xs-dx,ys-dy:,:] = up_corner_psf[-1:0:-1,:,:].copy()    #x<0, y>0
    if dy == 0:
        to_return[xs-dx:,:ys-dy,:] = up_corner_psf[:,::-1,:].copy()       #x>0, y<0
    else:
        to_return[xs-dx:,:ys-dy,:] = up_corner_psf[:,-1:0:-1,:].copy()    #x>0, y<0
    if (dx == 0) and (dy == 0):
        to_return[:xs-dx,:ys-dy,:] = up_corner_psf[::-1,::-1,:].copy()    #x<0,y<0
    elif (dx == 0) and (dy != 0):
        to_return[:xs-dx,:ys-dy,:] = up_corner_psf[::-1,-1:0:-1,:].copy() #x<0,y<0
    elif (dy == 0) and (dx != 0):
        to_return[:xs-dx,:ys-dy,:] = up_corner_psf[-1:0:-1,::-1,:].copy() #x<0,y<0
    else: #dx==1 and dy==1
        to_return[:xs-dx,:ys-dy,:] = up_corner_psf[-1:0:-1,-1:0:-1,:].copy()#x<0,y<0

    return to_return

def vec_to_halfvec(vec):
    """Transforms a vector np.arange(-N, M, dx) to np.arange(min(|vec|), max(N,M),dx)]"""
    d = vec[1:] - vec[:-1]
    if ((d/d.mean()).std() > 1e-14) or (d.mean() < 0):
        raise ValueError('vec must be np.arange() in increasing order')
    dx = d.mean()
    lowest = np.abs(vec).min()
    highest = np.abs(vec).max()
    return np.arange(lowest, highest + 0.1*dx, dx).astype(vec.dtype)
