import numpy as np

from cbamf import util
from cbamf import interpolation
from cbamf.comp import psfs, psfcalc

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
            pxsize=0.125, method='fftn', support_factor=2, normalize=False, sigkf=None,
            nkpts=None, cutoffval=None, *args, **kwargs):
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
            percentage of peak psf value to cutoff using a 3-axis exp(-(r-r0)**4)
            function where r0 is determined by cutoffval

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

        self.polychromatic = False
        self.sigkf = sigkf
        self.nkpts = nkpts
        self.cutoffval = cutoffval

        if self.sigkf is not None:
            self.nkpts = self.nkpts or 3
            self.polychromatic = True
        elif self.nkpts is not None:
            self.sigkf = 0.0
            self.polychromatic = True

        # FIXME -- zrange can't be none right now -- need to fix boundary calculations
        if zrange is None:
            zrange = (0, shape[0])
        self.zrange = zrange

        # text location of parameters for ease of extraction
        self.param_order = ['kfki', 'zslab', 'zscale', 'alpha', 'n2n1', 'laser_wavelength', 'sigkf']
        params = np.array( [ kfki,   zslab,   zscale,   alpha,   n2n1,   laser_wavelength,   sigkf ])
        self.param_dict = {k:params[i] for i,k in enumerate(self.param_order)}

        super(ExactLineScanConfocalPSF, self).__init__(*args, params=params,
                                                        shape=shape, **kwargs)

    def psf_slice(self, zint, size=11, zoffset=0.):
        """ Calculates the 3D psf at a particular z pixel height """
        tile = util.Tile(left=0, size=size, centered=True)
        z,y,x = tile.coords(meshed=False, flat=True)

        # calculate the current pixel value in 1/k, making sure we are above the slab
        zint = max(self._p2k(self._tz(zint)), 0)
        zoffset *= zint > 0

        x,y,z = [self._p2k(i) for i in [x,y,z+zoffset]]

        if self.polychromatic:
            psffunc = psfcalc.calculate_polychrome_linescan_psf
        else:
            psffunc = psfcalc.calculate_linescan_psf

        psf = psffunc(x, y, z, zint=zint, **self.args()).T
        return psf, tile.coords(meshed=True)

    def todict(self):
        return {k:self.params[i] for i,k in enumerate(self.param_order)}

    def args(self):
        """
        Pack the parameters into the form necessary for the integration
        routines above.  For example, packs for calculate_linescan_psf
        """
        d = self.todict()
        d.update({'polar_angle': self.polar_angle, 'normalize': self.normalize})
        d.pop('laser_wavelength')
        d.pop('zslab')
        d.pop('zscale')

        if not self.polychromatic:
            d.pop('sigkf')

        return d

    def _compatibility_patch(self):
        # FIXME -- why this function with __dict__.get? backwards compatibility
        # each of these parameters were added after the original class was made
        self.normalize = self.__dict__.get('normalize', False)
        self.cutoffval = self.__dict__.get('cutoffval', None)
        self.sigkf = self.__dict__.get('sigkf', None)
        self.nkpts = self.__dict__.get('nkpts', None)
        self.polychromatic = self.sigkf is not None or self.nkpts is not None

    def _p2k(self, v):
        """ Convert from pixel to 1/k_incoming (laser_wavelength/(2\pi)) units """
        return 2*np.pi*self.pxsize*v/self.param_dict['laser_wavelength']

    def _tz(self, z):
        """ Transform z to real-space coordinates from tile coordinates """
        return (z-self.param_dict['zslab'])*self.param_dict['zscale']

    def drift(self, z):
        """ Give the pixel offset at a given z value for the current parameters """
        return np.polyval(self.drift_poly, z)

    def measure_size_drift(self, z, size=21, zoffset=0.):
        """ Returns the 'size' of the psf in each direction a particular z (px) """
        psf, (z,y,x) = self.psf_slice(z, size=size, zoffset=zoffset)
        psf = psf / psf.sum()
        drift = moment(psf, z, order=1)
        size = [moment(psf, i, order=2) for i in (z,y,x)]
        return np.array(size), drift

    def characterize_psf(self):
        """ Get support size and drift polynomial for current set of params """
        l,u = max(self.zrange[0], self.param_dict['zslab']), self.zrange[1]

        size_l, drift_l = self.measure_size_drift(l, size=31)
        size_u, drift_u = self.measure_size_drift(u, size=31)

        # FIXME -- must be odd for now or have a better system for getting the center
        self.support = 2*self.support_factor*size_u.astype('int')+1
        self.drift_poly = np.polyfit([l, u], [drift_l, drift_u], 1)

    def get_params(self):
        return self.params

    def get_support_size(self, z=None):
        return self.support

    def update(self, params):
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

    def _kpad(self, field, finalshape, method='fftn', zpad=False):
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

        # FIXME -- need to normalize here?
        #kpsf /= (np.real(kpsf[0,0,0]) + 1e-15)
        return kpsf

    def execute(self, field):
        if any(field.shape != self.tile.shape):
            raise AttributeError("Field passed to PSF incorrect shape")

        outfield = np.zeros_like(field, dtype='float')
        zc,yc,xc = self.tile.coords(meshed=False, flat=True)

        for i,z in enumerate(zc):
            # in this loop, z in the index for self.slices and i is the field index
            # don't calculate this slice if we are outside the acceptable zrange
            if z < self.zrange[0] or z > self.zrange[1]:
                continue

            if i < self.support[0]/2 or i > self.tile.shape[0]-self.support[0]/2-1:
                continue

            # pad the psf slice for the convolution
            fs = np.array(self.tile.shape)
            fs[0] = self.support[0]
            kpsf = self._kpad(self.slices[i-self.zrange[0]], fs, method=self.method)

            # need to grab the right slice of the field to convolve with PSF
            zslice = np.s_[max(i-fs[0]/2,0):min(i+fs[0]/2+1,self.tile.shape[0])]

            kfield = self.fftn(field[zslice])

            if self.method == 'fftn':
                outfield[i] = np.real(self.ifftn(kfield * kpsf))[self.support[0]/2]
            else:
                outfield[i] = np.real(self.ifftn(kfield * kpsf)).sum(axis=0)

        return outfield

    def _setup_ffts(self):
        if psfs.hasfftw and self.method == 'fftn':
            # adjust the shape for the transforms we are really doing
            shape = self.tile.shape.copy()
            shape[0] = self.support[0]

            self._fftn_data = psfs.pyfftw.n_byte_align_empty(shape, 16, dtype='double')
            self._fftn = psfs.rfftn(self._fftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level)

            t = np.zeros(shape)
            o = self.fftn(t)
            self._ifftn_data = psfs.pyfftw.n_byte_align_empty(o.shape, 16, dtype='complex')
            self._ifftn = psfs.irfftn(self._ifftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level)

        elif psfs.hasfftw and self.method == 'fft2':
            shape = self.tile.shape.copy()
            shape[0] = self.support[0]

            self._fftn_data = psfs.pyfftw.n_byte_align_empty(shape, 16, dtype='double')
            self._fftn = psfs.rfft2(self._fftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level)

            t = np.zeros(shape)
            o = self.fftn(t)
            self._ifftn_data = psfs.pyfftw.n_byte_align_empty(o.shape, 16, dtype='complex')
            self._ifftn = psfs.irfft2(self._ifftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level)

    def __getstate__(self):
        odict = super(ChebyshevLineScanConfocalPSF, self).__getstate__()
        util.cdd(odict, ['slices'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self._compatibility_patch()

class ChebyshevLineScanConfocalPSF(ExactLineScanConfocalPSF):
    def __init__(self, cheb_degree=3, cheb_evals=4, *args, **kwargs):
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
        zc,yc,xc = self.tile.coords(meshed=False, flat=True)

        kfield = self.fftn(field)
        for k,c in enumerate(self.cheb.coefficients):
            pad = self._kpad(c, finalshape=self.tile.shape, zpad=True)
            cov = np.real(self.ifftn(kfield * pad))

            outfield += self.cheb.tk(k, zc)[:,None,None] * cov

        return outfield

    def _setup_ffts(self):
        if psfs.hasfftw:
            shape = self.tile.shape.copy()
            self._fftn_data = psfs.pyfftw.n_byte_align_empty(shape, 16, dtype='double')
            self._fftn = psfs.rfftn(self._fftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level)

            oshape = self.fftn(np.zeros(shape)).shape
            self._ifftn_data = psfs.pyfftw.n_byte_align_empty(oshape, 16, dtype='complex')
            self._ifftn = psfs.irfftn(self._ifftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level)

    def __getstate__(self):
        odict = self.__dict__.copy()
        util.cdd(odict, ['_rx', '_ry', '_rz', '_rvecs', '_rlen'])
        util.cdd(odict, ['_kx', '_ky', '_kz', '_kvecs', '_klen'])
        util.cdd(odict, ['_fftn', '_ifftn', '_fftn_data', '_ifftn_data'])
        util.cdd(odict, ['_memoize_clear', '_memoize_caches'])
        util.cdd(odict, ['rpsf', 'kpsf'])
        util.cdd(odict, ['cheb'])
        return odict
