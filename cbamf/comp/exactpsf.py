import warnings
import numpy as np
import scipy.ndimage as nd

from cbamf import util
from cbamf import interpolation
from cbamf.comp import psfs, psfcalc

def calc_quintile(p, quintile, axis=0):
    """p = rho, v = xyz"""
    inds = [0,1,2]
    inds.remove(axis)
    partial_int = np.array([0] + p.sum(axis=tuple(inds)).tolist() + [0])
    lut = 0*partial_int
    for a in xrange(lut.size):
        lut[a] = np.trapz(partial_int[:a+1])
    just_above = np.nonzero(lut > quintile)[0][0]
    just_below = just_above - 1
    slope = lut[just_above] - lut[just_below]
    delta = (quintile-lut[just_below]) / slope
    ans_px = just_below + delta - 1 #-1 for the [0] + ... + [0] at the ends
    ans_coord = ans_px - p.shape[axis]/2
    return ans_coord

def median_width(p, axis=0, order=1):
    """
    Rather than calculating a moment, calculates a median (order=1)
    and the max distance to the upper or lower quartiles (order=2)
    """
    if order == 1:
        return calc_quintile(p, 0.5, axis=axis)
    elif order == 2:
        med = calc_quintile(p, 0.50, axis=axis)
        low = calc_quintile(p, 0.25, axis=axis)
        hih = calc_quintile(p, 0.75, axis=axis)
        return np.max([hih-med, med-low])

def moment(p, v, order=1):
    """ Calculates the moments of the probability distribution p with vector v """
    if order == 1:
        return (v*p).sum()
    elif order == 2:
        return np.sqrt( ((v**2)*p).sum() - (v*p).sum()**2 )

#=============================================================================
# The actual interfaces that can be used in the cbamf system
#=============================================================================
class ExactLineScanConfocalPSF(psfs.PSF):
    def __init__(self, shape, zrange, laser_wavelength=0.488, zslab=0.,
            zscale=1.0, kfki=0.889, n2n1=1.44/1.518, alpha=1.173, polar_angle=0.,
            pxsize=0.125, method='fftn', support_factor=2, normalize=False, sigkf=0.0,
            nkpts=None, cutoffval=None, measurement_iterations=None,
            k_dist='gaussian', use_J1=True, sph6_ab=None, scale_fix=True,
            cutbyval=True, cutfallrate=0.25, cutedgeval=1e-12, use_laggauss=True, 
            pinhole_width=None, do_pinhole=False, *args, **kwargs):
        """
        PSF for line-scanning confocal microscopes that can be used with the
        cbamf framework.  Calculates the spatially varying point spread
        function for confocal microscopes and allows them to be applied to
        images as a convolution.

        This PSF assumes that the z extent is large compared to the image size
        and so calculates the local PSF for every z layer in the image.

        Parameters:
        -----------
        shape : tuple
            Shape of the image in (z,y,x) pixel numbers (to be deprecated)

        zrange : tuple
            range of z pixels over which we should calculate the psf, good pixels
            being zrange[0] <= z <= zrange[1]. currently must be set to the interior
            of the image, so [state.pad, state.image.shape[0] - state.pad]

        laser_wavelength : float
            wavelength of light in um of the incoming laser light

        zslab : float
            Pixel position of the optical interface where 0 is the edge of the
            image in the z direction

        zscale : float
            Scaling of the z pixel size, so that each pixel is located at
            zscale*(z - zint), such that the interface does not move with zscale

        kfki : float
            Ratio of outgoing to incoming light wavevectors, 2\pi/\lambda

        n2n1 : float
            Ratio of the index mismatch across the optical interface. For typical
            glass with glycerol/water 80/20 mixture this is 1.4/1.518

        alpha : float
            Aperture of the lens in radians, set by arcsin(n2n1)?

        polar_angle : float
            the angle of the light polarization with respect to the scanning axis

        pxsize : float
            the size of a xy pixel in um, defaults to cohen group size 0.125 um

        method : str
            Either ['fftn', 'fft2'] which represent the way the convolution is
            performed.  Currently 'fftn' is recommended - while slower is more
            accurate

        support_factor : integer
            size of the support

        normalize : boolean
            if True, use the normalization as calculated by the PSF instead of
            unit normalization

        sigkf : float
            Width of wavelengths to use in the polychromatic psf, None is
            monochromatic. Values for kfki = kfki +- sigkf, unitless

        nkpts : integer
            number of integration points to use for the polychromatic psf

        cutoffval : float
            Percentage of peak psf value to cutoff using a 3-axis
            exp(-(r-r0)**4) function where r0 is determined by cutoffval. A
            good value for this is the bit depth of the camera, or possibly the
            SNR, so 1/2**8 or 1/50.

        measurement_iterations : int
            number of interations used when trying to find the center of mass
            of the psf in a certain slice

        k_dist : str
            Eithe ['gaussian', 'gamma'] which control the wavevector
            distribution for the polychromatic detection psf. Default
            is Gaussian. 
            
        use_J1 : boolean
            Which hdet confocal model to use. Set to True to include the 
            J1 term corresponding to a large-NA focusing lens, False to
            exclude it. Default is True
            
        scale_fix : boolean
            fix previous issue with zscale no coupled to actual calculation
            besides through zint

        cutbyval : boolean
            If True, cuts the PSF based on the actual value instead of the
            position associated with the nearest value.

        cutfallrate : float
            The relative value of the cutoffval over which to damp the
            remaining values of the psf. 0.3 seems a good fit now.

        cutedgeval : float
            The value with which to determine the edge of the psf, typically
            taken around floating point, 1e-12
            
        use_laggauss : Bool
            Whether to use the old/inaccurate Gauss-Hermite quadrature for
            the line integral, or a x=sinh(a*t) rule and Gauss-Laguerre
            quadrature (more accurate). Default is True. 

        pinhole_width : Float
            The width of the line illumination, in 1/k units. Default is 1.0.

        do_pinhole : Bool
            Whether or not to include pinhole line width in the sampling. 
            Default is False. 

        Notes:
            a = ExactLineScanConfocalPSF((64,)*3)
            psf, (z,y,x) = a.psf_slice(1., size=51)
            imshow((psf*r**4)[:,:,25], cmap='bone')
        """
        self.pxsize = pxsize
        self.method = method
        self.polar_angle = polar_angle
        self.support_factor = support_factor
        self.normalize = normalize
        self.measurement_iterations = measurement_iterations or 1

        self.polychromatic = False
        self.sigkf = sigkf
        self.nkpts = nkpts
        self.cutoffval = cutoffval
        self.scale_fix = scale_fix
        self.cutbyval = cutbyval
        self.cutfallrate = cutfallrate
        self.cutedgeval = cutedgeval
        
        self.k_dist = k_dist
        self.use_J1 = use_J1

        self.use_laggauss = use_laggauss
        self.do_pinhole = do_pinhole

        if self.sigkf is not None:
            self.nkpts = self.nkpts or 3
            self.polychromatic = True
        elif self.nkpts is not None:
            self.sigkf = 0.0
            self.polychromatic = True
        else:
            self.sigkf = sigkf = 0.0
            self.polychromatic = False
            
        if (sph6_ab is not None) and (not np.isnan(sph6_ab)):
            self.use_sph6_ab = True #necessary? FIXME
        else:
            self.use_sph6_ab = False
            
        if (pinhole_width is not None) or do_pinhole:
            self.num_line_pts = 3
        else:
            self.num_line_pts = 1
        pinhole_width = pinhole_width if (pinhole_width is not None) else 1.0

        # FIXME -- zrange can't be none right now -- need to fix boundary calculations
        if zrange is None:
            zrange = (0, shape[0])
        self.zrange = zrange

        # text location of parameters for ease of extraction
        self.param_order = ['kfki', 'zslab', 'zscale', 'alpha', 'n2n1', 'laser_wavelength', 'sigkf', 'sph6_ab', 'pinhole_width']
        params = np.array( [ kfki,   zslab,   zscale,   alpha,   n2n1,   laser_wavelength,   sigkf, sph6_ab, pinhole_width ])

        # the next statements must occur in the correct order so that
        # other parameters are not deleted by mistake
        if not self.polychromatic:
            self.param_order.pop(-3)
            params = np.delete(params, -3)

        if not self.use_sph6_ab:
            self.param_order.pop(-2)
            params = np.delete(params, -2)
            
        if not self.do_pinhole:
            self.param_order.pop(-1)
            params = np.delete(params, -1)

        self.param_dict = {k:params[i] for i,k in enumerate(self.param_order)}

        super(ExactLineScanConfocalPSF, self).__init__(*args, params=params,
                                                        shape=shape, **kwargs)

    def psf_slice(self, zint, size=11, zoffset=0., getextent=False):
        """
        Calculates the 3D psf at a particular z pixel height

        Parameters:
        -----------
        zint : float
            z pixel height in image coordinates , converted to 1/k by the
            function using the slab position as well

        size : int, list, tuple
            The size over which to calculate the psf, can be 1 or 3 elements
            for the different axes in image pixel coordinates

        zoffset : float
            Offset in pixel units to use in the calculation of the psf

        cutval : float
            If not None, the psf will be cut along a curve corresponding to
            p(r) == 0 with exponential damping exp(-d^4)

        getextent : boolean
            If True, also return the extent of the psf in pixels for example
            to get the support size. Can only be used with cutval.
        """
        # calculate the current pixel value in 1/k, making sure we are above the slab
        zint = max(self._p2k(self._tz(zint)), 0)
        offset = np.array([zoffset*(zint>0), 0, 0])
        
        if self.scale_fix:
            scale = [self.param_dict['zscale'], 1.0, 1.0]
        else:
            scale = [1.0]*3

        # create the coordinate vectors for where to actually calculate the 
        tile = util.Tile(left=0, size=size, centered=True)
        vecs = tile.coords(form='flat')
        vecs = [self._p2k(s*i+o) for i,s,o in zip(vecs, scale, offset)]

        if self.polychromatic:
            psffunc = psfcalc.calculate_polychrome_linescan_psf
        else:
            psffunc = psfcalc.calculate_linescan_psf

        psf = psffunc(*vecs[::-1], zint=zint, **self.args()).T
        vec = tile.coords(form='meshed')

        # create a smoothly varying point spread function by cutting off the psf
        # at a certain value and smoothly taking it to zero
        if self.cutoffval is not None and not self.cutbyval:
            # find the edges of the PSF
            edge = psf > psf.max() * self.cutoffval
            dd = nd.morphology.distance_transform_edt(~edge)

            # calculate the new PSF and normalize it to the new support
            psf = psf * np.exp(-dd**4)
            psf /= psf.sum()

            if getextent:
                # the size is determined by the edge plus a 2 pad for the
                # exponential damping to zero at the edge
                size = np.array([
                    (vec*edge).min(axis=(1,2,3))-2,
                    (vec*edge).max(axis=(1,2,3))+2,
                ]).T
                return psf, vec, size
            return psf, vec

        # perform a cut by value instead
        if self.cutoffval is not None and self.cutbyval:
            cutval = self.cutoffval * psf.max()

            dd = (psf - cutval) / cutval
            dd[dd > 0] = 0.

            # calculate the new PSF and normalize it to the new support
            psf = psf * np.exp(-(dd / self.cutfallrate)**4)
            psf /= psf.sum()

            # let the small values determine the edges
            edge = psf > cutval * self.cutedgeval
            if getextent:
                # the size is determined by the edge plus a 2 pad for the
                # exponential damping to zero at the edge
                size = np.array([
                    (vec*edge).min(axis=(1,2,3))-2,
                    (vec*edge).max(axis=(1,2,3))+2,
                ]).T
                return psf, vec, size
            return psf, vec

        return psf, vec

    def todict(self):
        return {k:self.params[i] for i,k in enumerate(self.param_order)}

    def args(self):
        """
        Pack the parameters into the form necessary for the integration
        routines above.  For example, packs for calculate_linescan_psf
        """
        d = self.todict()
        d.update({'polar_angle': self.polar_angle, 'normalize': self.normalize,
                'include_K3_det':self.use_J1}) 
        if self.polychromatic:
            d.update({'k_dist':self.k_dist})
        d.pop('laser_wavelength')
        d.pop('zslab')
        d.pop('zscale')

        if not self.polychromatic and d.has_key('sigkf'):
            d.pop('sigkf')
        if not self.use_sph6_ab and d.has_key('sph6_ab'):
            d.pop('sph6_ab')
            
        if self.do_pinhole:
            d.update({'nlpts':self.num_line_pts})
        
        if self.use_laggauss:
            d.update({'use_laggauss':self.use_laggauss})

        return d

    def _compatibility_patch(self):
        # FIXME -- why this function with __dict__.get? backwards compatibility
        # each of these parameters were added after the original class was made
        self.normalize = self.__dict__.get('normalize', False)
        self.cutoffval = self.__dict__.get('cutoffval', None)
        self.cutbyval = self.__dict__.get('cutbyval', False)
        self.cutfallrate = self.__dict__.get('cutfallrate', 0.25)
        self.cutedgeval = self.__dict__.get('cutedgeval', 1e-12)
        self.sigkf = self.__dict__.get('sigkf', None)
        self.nkpts = self.__dict__.get('nkpts', None)
        self.polychromatic = self.sigkf is not None or self.nkpts is not None
        self.measurement_iterations = self.__dict__.get('measurement_iterations', 1)
        self.k_dist = self.__dict__.get('k_dist', 'gaussian')
        self.use_J1 = self.__dict__.get('use_J1', True)
        self.use_sph6_ab = self.__dict__.get('use_sph6_ab', False)
        self.scale_fix = self.__dict__.get('scale_fix', False)
        self.use_laggauss = self.__dict__.get('use_laggauss', False)
        self.do_pinhole = self.__dict__.get('do_pinhole', False)
        self.num_line_pts = self.__dict__.get('num_line_pts', 1)
        

    def _p2k(self, v):
        """ Convert from pixel to 1/k_incoming (laser_wavelength/(2\pi)) units """
        return 2*np.pi*self.pxsize*v/self.param_dict['laser_wavelength']

    def _tz(self, z):
        """ Transform z to real-space coordinates from tile coordinates """
        return (z-self.param_dict['zslab'])*self.param_dict['zscale']

    def drift(self, z):
        """ Give the pixel offset at a given z value for the current parameters """
        return np.polyval(self.drift_poly, z)

    def measure_size_drift(self, z, size=31, zoffset=0.):
        """ Returns the 'size' of the psf in each direction a particular z (px) """
        drift = 0.0
        for i in xrange(self.measurement_iterations):
            psf, vec = self.psf_slice(z, size=size, zoffset=zoffset+drift)
            psf = psf / psf.sum()

            drift += moment(psf, vec[0], order=1)
            psize = [moment(psf, j, order=2) for j in vec]
        return np.array(psize), drift

    def characterize_psf(self):
        """ Get support size and drift polynomial for current set of params """
        l,u = max(self.zrange[0], self.param_dict['zslab']), self.zrange[1]

        size_l, drift_l = self.measure_size_drift(l)
        size_u, drift_u = self.measure_size_drift(u)

        # FIXME -- must be odd for now or have a better system for getting the center
        self.support = util.oddify(2*self.support_factor*size_u.astype('int'))
        self.drift_poly = np.polyfit([l, u], [drift_l, drift_u], 1)

        if self.cutoffval is not None:
            psf, vec, size_l = self.psf_slice(l, size=51, zoffset=drift_l, getextent=True)
            psf, vec, size_u = self.psf_slice(u, size=51, zoffset=drift_u, getextent=True)

            ss = [np.abs(i).sum(axis=-1) for i in [size_l, size_u]]
            self.support = util.oddify(util.amax(*ss))

    def get_params(self):
        return self.params

    def get_support_size(self, z=None):
        return self.support

    def update(self, params):
        #Clipping params to computable values:
        alpha_ind = self.param_order.index('alpha')
        zscal_ind = self.param_order.index('zscale')
        max_alpha = np.pi*0.5
        max_zscle = 100
        if params[alpha_ind] < 1e-3 or params[alpha_ind] > max_alpha:
            warnings.warn('Invalid alpha, clipping', RuntimeWarning)
            params[alpha_ind] = np.clip(params[alpha_ind], 1e-3, max_alpha-1e-3)
        if params[zscal_ind] < 1e-3 or params[zscal_ind] > max_zscle:
            warnings.warn('Invalid zscale, clipping', RuntimeWarning)
            params[zscal_ind] = np.clip(params[zscal_ind], 1e-3, max_zscle-1e-3)
        
        self.params[:] = params[:]
        self.param_dict = self.todict()
        self.characterize_psf()

        self.slices = []
        for i in xrange(self.zrange[0], self.zrange[1]+1):
            zdrift = self.drift(i)
            psf, vec = self.psf_slice(i, size=self.support, zoffset=zdrift)
            self.slices.append(psf)

        self.slices = np.array(self.slices)

    def set_tile(self, tile):
        if (self.tile.shape != tile.shape).any():
            self.tile = tile
            self._setup_ffts()

    def _kpad(self, field, finalshape, method='fftn', zpad=False, norm=True):
        """
        fftshift and pad the field with zeros until it has size finalshape.
        if zpad is off, then no padding is put on the z direction. returns
        the fourier transform of the field
        """
        currshape = np.array(field.shape)

        if any(finalshape < currshape):
            raise IndexError("PSF tile size is less than minimum support size")

        d = finalshape - currshape

        # fix off-by-one issues when going odd to even tile sizes
        o = d % 2
        d /= 2

        if not zpad:
            o[0] = 0

        axes = None if method == 'fftn' else (1,2)
        pad = tuple((d[i]+o[i],d[i]) for i in [0,1,2])
        rpsf = np.pad(field, pad, mode='constant', constant_values=0)
        rpsf = np.fft.ifftshift(rpsf, axes=axes)
        kpsf = self.fftn(rpsf)

        if norm:
            kpsf[0,0,0] = 1.0
        return kpsf

    def execute(self, field):
        if any(field.shape != self.tile.shape):
            raise AttributeError("Field passed to PSF incorrect shape")

        outfield = np.zeros_like(field, dtype='float')
        zc,yc,xc = self.tile.coords(form='flat')

        # here's the plan. we are going to rotate the field so that the current
        # plane of interest is in the center. we then crop the image to the
        # size of the support so that we are only convolving a small region.
        # finally, take the mid plane back out as the solution.
        for i,z in enumerate(zc):
            # pad the psf slice for the convolution
            fs = np.array(self.tile.shape)
            fs[0] = self.support[0]

            if z < self.zrange[0] or z > self.zrange[1]:
                continue

            zslice = int(np.clip(z, *self.zrange) - self.zrange[0])
            middle = field.shape[0]/2

            subpsf = self._kpad(self.slices[zslice], fs, method=self.method, norm=True)
            subfield = np.roll(field, middle - i, axis=0)
            subfield = subfield[middle-fs[0]/2:middle+fs[0]/2+1]

            kfield = self.fftn(subfield)

            if self.method == 'fftn':
                outfield[i] = np.real(self.ifftn(kfield * subpsf))[self.support[0]/2]
            else:
                outfield[i] = np.real(self.ifftn(kfield * subpsf)).sum(axis=0)

        return outfield

    def _setup_ffts(self):
        if psfs.hasfftw and self.method == 'fftn':
            # adjust the shape for the transforms we are really doing
            shape = self.tile.shape.copy()
            shape[0] = self.support[0]

            self._fftn_data = psfs.pyfftw.n_byte_align_empty(shape, 16, dtype='double')
            self._fftn = psfs.rfftn(self._fftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level, s=shape)

            oshape = self.fftn(np.zeros(shape)).shape
            self._ifftn_data = psfs.pyfftw.n_byte_align_empty(oshape, 16, dtype='complex')
            self._ifftn = psfs.irfftn(self._ifftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level, s=shape)

        elif psfs.hasfftw and self.method == 'fft2':
            shape = self.tile.shape.copy()
            shape[0] = self.support[0]

            self._fftn_data = psfs.pyfftw.n_byte_align_empty(shape, 16, dtype='double')
            self._fftn = psfs.rfft2(self._fftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level, s=shape)

            oshape = self.fftn(np.zeros(shape)).shape
            self._ifftn_data = psfs.pyfftw.n_byte_align_empty(oshape, 16, dtype='complex')
            self._ifftn = psfs.irfft2(self._ifftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level, s=shape)

    def __getstate__(self):
        odict = self.__dict__.copy()
        util.cdd(odict, ['_rx', '_ry', '_rz', '_rvecs', '_rlen'])
        util.cdd(odict, ['_kx', '_ky', '_kz', '_kvecs', '_klen'])
        util.cdd(odict, ['_fftn', '_ifftn', '_fftn_data', '_ifftn_data'])
        util.cdd(odict, ['_memoize_clear', '_memoize_caches'])
        util.cdd(odict, ['rpsf', 'kpsf'])
        util.cdd(odict, ['cheb', 'slices'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self._compatibility_patch()
        self._setup_ffts()

class ChebyshevLineScanConfocalPSF(ExactLineScanConfocalPSF):
    def __init__(self, cheb_degree=6, cheb_evals=8, *args, **kwargs):
        """
        Same as ExactLineScanConfocalPSF except that integration is performed
        in the 4th dimension by employing fast Chebyshev approximates to how
        the PSF varies with depth into the sample. For help, see
        ExactLineScanConfocalPSF.

        Additional parameters:
        ----------------------
        cheb_degree : integer
            degree of the Chebyshev approximant

        cheb_evals : integer
            number of interpolation points used to create the coefficient matrix
        """
        self.cheb_degree = cheb_degree
        self.cheb_evals = cheb_evals

        # make sure that we are use the parent class 'fftn' method
        kwargs.setdefault('method', 'fftn')
        super(ChebyshevLineScanConfocalPSF, self).__init__(*args, **kwargs)

    def update(self, params):
        #Clipping params to computable values:
        alpha_ind = self.param_order.index('alpha')
        zscal_ind = self.param_order.index('zscale')
        max_alpha = np.pi*0.5
        max_zscle = 100
        if params[alpha_ind] < 1e-3 or params[alpha_ind] > max_alpha:
            warnings.warn('Invalid alpha, clipping', RuntimeWarning)
            params[alpha_ind] = np.clip(params[alpha_ind], 1e-3, max_alpha-1e-3)
        if params[zscal_ind] < 1e-3 or params[zscal_ind] > max_zscle:
            warnings.warn('Invalid zscale, clipping', RuntimeWarning)
            params[zscal_ind] = np.clip(params[zscal_ind], 1e-3, max_zscle-1e-3)
        
        self.params[:] = params[:]
        self.param_dict = self.todict()
        self.characterize_psf()

        self.cheb = interpolation.ChebyshevInterpolation1D(self.psf, window=self.zrange,
                        degree=self.cheb_degree, evalpts=self.cheb_evals)

    def psf(self, z):
        psf = []
        for i in z:
            p, _ = self.psf_slice(i, size=self.support, zoffset=self.drift(i))
            psf.append(p)
        return np.rollaxis(np.array(psf), 0, 4)

    def execute(self, field):
        if any(field.shape != self.tile.shape):
            raise AttributeError("Field passed to PSF incorrect shape")

        outfield = np.zeros_like(field, dtype='float')
        zc,yc,xc = self.tile.coords(form='flat')

        kfield = self.fftn(field)
        for k,c in enumerate(self.cheb.coefficients):
            pad = self._kpad(c, finalshape=self.tile.shape, zpad=True, norm=False)
            cov = np.real(self.ifftn(kfield * pad))

            outfield += self.cheb.tk(k, zc)[:,None,None] * cov

        return outfield

    def _setup_ffts(self):
        if psfs.hasfftw:
            shape = self.tile.shape.copy()
            self._fftn_data = psfs.pyfftw.n_byte_align_empty(shape, 16, dtype='double')
            self._fftn = psfs.rfftn(self._fftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level, s=shape)

            oshape = self.fftn(np.zeros(shape)).shape
            self._ifftn_data = psfs.pyfftw.n_byte_align_empty(oshape, 16, dtype='complex')
            self._ifftn = psfs.irfftn(self._ifftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level, s=shape)

    def __getstate__(self):
        odict = self.__dict__.copy()
        util.cdd(odict, ['_rx', '_ry', '_rz', '_rvecs', '_rlen'])
        util.cdd(odict, ['_kx', '_ky', '_kz', '_kvecs', '_klen'])
        util.cdd(odict, ['_fftn', '_ifftn', '_fftn_data', '_ifftn_data'])
        util.cdd(odict, ['_memoize_clear', '_memoize_caches'])
        util.cdd(odict, ['rpsf', 'kpsf'])
        util.cdd(odict, ['cheb'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self._compatibility_patch()
        self._setup_ffts()

class FixedSSChebLinePSF(ChebyshevLineScanConfocalPSF):
    def __init__(self, support_size=[35,17,25], *args, **kwargs):
        self.cutoffval = None
        self.support = np.array(support_size)
        super(FixedSSChebLinePSF, self).__init__(*args, **kwargs)
        
    def characterize_psf(self):
        """ Get support size and drift polynomial for current set of params """
        l,u = max(self.zrange[0], self.param_dict['zslab']), self.zrange[1]

        size_l, drift_l = self.measure_size_drift(l)
        size_u, drift_u = self.measure_size_drift(u)

        # FIXME -- must be odd for now or have a better system for getting the center
        # self.support = util.oddify(2*self.support_factor*size_u.astype('int'))
        self.drift_poly = np.polyfit([l, u], [drift_l, drift_u], 1)
        
    def _compatibility_patch(self):
        self.support = self.__dict__.get('support', np.array([35,17,25]))
        super(FixedSSChebLinePSF, self)._compatibility_patch()

class FixedBigSSChebLinePSF(FixedSSChebLinePSF):
    """
    PSF with a bigger fixed global support size of [61, 25, 33]
    """
    def characterize_psf(self):
        """ Get support size and drift polynomial for current set of params """
        l,u = max(self.zrange[0], self.param_dict['zslab']), self.zrange[1]

        size_l, drift_l = self.measure_size_drift(l)
        size_u, drift_u = self.measure_size_drift(u)

        self.support = np.array([61, 25, 33])
        self.drift_poly = np.polyfit([l, u], [drift_l, drift_u], 1)
