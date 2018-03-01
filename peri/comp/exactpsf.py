from builtins import range
from future.utils import iteritems

import warnings
import numpy as np
import scipy.ndimage as nd

from collections import OrderedDict

from peri import util, interpolation
from peri.comp import psfs, psfcalc
from peri.fft import fft, fftkwargs

def moment(p, v, order=1):
    """ Calculates the moments of the probability distribution p with vector v """
    if order == 1:
        return (v*p).sum()
    elif order == 2:
        return np.sqrt( ((v**2)*p).sum() - (v*p).sum()**2 )

#=============================================================================
# The actual interfaces that can be used in the peri system
#=============================================================================
class ExactPSF(psfs.PSF):
    def __init__(self, shape=None, zrange=None, laser_wavelength=0.488,
            zslab=0., zscale=1.0, kfki=0.889, n2n1=1.44/1.518, alpha=1.173,
            polar_angle=0., pxsize=0.125, support_factor=2, normalize=False,
            sigkf=0.0, nkpts=None, cutoffval=None, measurement_iterations=None,
            k_dist='gaussian', use_J1=True, sph6_ab=None, global_zscale=False,
            cutbyval=False, cutfallrate=0.25, cutedgeval=1e-12,
            pinhole_width=None, do_pinhole=False, *args, **kwargs):
        """
        Superclass for all the exact PSFs, i.e. any PSF that is based on
        physical properties of the imaging system such as the laser
        wavelength.

        This PSF functions by calculating the local PSF for every z layer
        in the image, and convolving each layer independently (numerically
        the exact model of image formation).

        Parameters
        ----------
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

        cutbyval : boolean
            If True, cuts the PSF based on the actual value instead of the
            position associated with the nearest value.

        cutfallrate : float
            The relative value of the cutoffval over which to damp the
            remaining values of the psf. 0.3 seems a good fit now.

        cutedgeval : float
            The value with which to determine the edge of the psf, typically
            taken around floating point, 1e-12

        pinhole_width : Float
            The width of the line illumination, in 1/k units. Default is 1.0.

        do_pinhole : Bool
            Whether or not to include pinhole line width in the sampling.
            Default is False.
        """
        self.pxsize = pxsize
        self.polar_angle = polar_angle
        self.support_factor = support_factor
        self.normalize = normalize
        self.measurement_iterations = measurement_iterations or 11
        self.global_zscale = global_zscale

        self.polychromatic = False
        self.sigkf = sigkf
        self.nkpts = nkpts
        self.cutoffval = cutoffval
        self.cutbyval = cutbyval
        self.cutfallrate = cutfallrate
        self.cutedgeval = cutedgeval

        self.k_dist = k_dist
        self.use_J1 = use_J1

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

        if shape and zrange is None:
            zrange = (0, shape.shape[0])
        self.zrange = zrange

        # text location of parameters for ease of extraction
        params = [
            'kfki', 'zslab', 'zscale', 'alpha', 'n2n1', 'laser-wavelength',
            'sigkf', 'sph6-ab'
        ]
        values = np.array([
            kfki,   zslab,   zscale,   alpha,   n2n1,   laser_wavelength,
            sigkf,    sph6_ab
        ])

        # the next statements must occur in the correct order so that
        # other parameters are not deleted by mistake
        if not self.polychromatic:
            ind = params.index('sigkf')
            params.pop(ind)
            values = np.delete(values, ind)

        if not self.use_sph6_ab:
            ind = params.index('sph6-ab')
            params.pop(ind)
            values = np.delete(values, ind)

        # if not self.do_pinhole:  #this canbe appended if self.do_pinhole
            # ind = params.index('pinhole-width')
            # params.pop(ind)
            # values = np.delete(values, ind)

        for i in range(len(params)):
            if params[i] is 'zscale' and self.global_zscale:
                continue
            params[i] = 'psf-' + params[i]

        super(ExactPSF, self).__init__(
            *args, params=params, values=values, shape=shape, **kwargs
        )

    def psffunc(self, *args, **kwargs):
        """
        The function to evaluate the exact psf. Syntax must be:
        
            psf = psffunc(x, y, z, \**kwargs)

        and return a ``[x.size, y.size, z.size]`` numpy.ndarray, where x,y,z
        are 1D arrays.
        Implement in subclass.
        """
        raise NotImplementedError('Supply psffunc in subclass')

    @property
    def zscale(self):
        return 'zscale' if self.global_zscale else 'psf-zscale'

    def set_shape(self, shape, inner):
        if self.zrange is None:
            self.zrange = (0, shape.shape[0])
        super(ExactPSF, self).set_shape(shape, inner)

    def psf_slice(self, zint, size=11, zoffset=0., getextent=False):
        """
        Calculates the 3D psf at a particular z pixel height

        Parameters
        ----------
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
        scale = [self.param_dict[self.zscale], 1.0, 1.0]

        # create the coordinate vectors for where to actually calculate the
        tile = util.Tile(left=0, size=size, centered=True)
        vecs = tile.coords(form='flat')
        vecs = [self._p2k(s*i+o) for i,s,o in zip(vecs, scale, offset)]

        psf = self.psffunc(*vecs[::-1], zint=zint, **self.pack_args()).T
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
        return {k:self.params[i] for i,k in enumerate(self.params)}

    def pack_args(self):
        """
        Pack the parameters into the form necessary for the integration
        routines above in psfcalc.
        """
        raise NotImplementedError('Implement in Subclass')

    def _p2k(self, v):
        """ Convert from pixel to 1/k_incoming (laser_wavelength/(2\pi)) units """
        return 2*np.pi*self.pxsize*v/self.param_dict['psf-laser-wavelength']

    def _tz(self, z):
        """ Transform z to real-space coordinates from tile coordinates """
        return (z-self.param_dict['psf-zslab'])*self.param_dict[self.zscale]

    def drift(self, z):
        """ Give the pixel offset at a given z value for the current parameters """
        return np.polyval(self.drift_poly, z)

    def measure_size_drift(self, z, size=31, zoffset=0.):
        """ Returns the 'size' of the psf in each direction a particular z (px) """
        drift = 0.0
        for i in range(self.measurement_iterations):
            psf, vec = self.psf_slice(z, size=size, zoffset=zoffset+drift)
            psf = psf / psf.sum()

            drift += moment(psf, vec[0], order=1)
            psize = [moment(psf, j, order=2) for j in vec]
        return np.array(psize), drift

    def characterize_psf(self):
        """ Get support size and drift polynomial for current set of params """
        # there may be an issue with the support and characterization--
        # it might be best to do the characterization with the same support
        # as the calculated psf.
        l,u = max(self.zrange[0], self.param_dict['psf-zslab']), self.zrange[1]

        size_l, drift_l = self.measure_size_drift(l)
        size_u, drift_u = self.measure_size_drift(u)

        # must be odd for now or have a better system for getting the center
        self.support = util.oddify(2*self.support_factor*size_u.astype('int'))
        self.drift_poly = np.polyfit([l, u], [drift_l, drift_u], 1)

        if self.cutoffval is not None:
            psf, vec, size_l = self.psf_slice(l, size=51, zoffset=drift_l, getextent=True)
            psf, vec, size_u = self.psf_slice(u, size=51, zoffset=drift_u, getextent=True)

            ss = [np.abs(i).sum(axis=-1) for i in [size_l, size_u]]
            self.support = util.oddify(util.amax(*ss))

    def get_padding_size(self, tile, z=None):
        return util.Tile(self.support)

    def update(self, params, values):
        self.update_values(params, values)
        self.characterize_psf()

        self.slices = []
        for i in range(self.zrange[0], self.zrange[1]+1):
            zdrift = self.drift(i)
            psf, vec = self.psf_slice(i, size=self.support, zoffset=zdrift)
            self.slices.append(psf)

        self.slices = np.array(self.slices)
        return True

    def update_values(self, params, values):
        self.set_values(params, values)

        #Clipping params to computable values:
        alpha = self.param_dict['psf-alpha']
        zscale = self.param_dict[self.zscale]
        wavelength = self.param_dict['psf-laser-wavelength']
        max_alpha, max_zscale = np.pi/2, 100.

        if alpha < 1e-3 or alpha > max_alpha:
            warnings.warn('Invalid alpha, clipping', RuntimeWarning)
            self.param_dict['psf-alpha'] = np.clip(alpha, 1e-3, max_alpha-1e-3)
        if zscale < 1e-3 or zscale > max_zscale:
            warnings.warn('Invalid zscale, clipping', RuntimeWarning)
            self.param_dict[self.zscale] = np.clip(zscale, 1e-3, max_zscale-1e-3)
        if wavelength < 1e-3:
            warnings.warn('Invalid laser wavelength, clipping', RuntimeWarning)
            self.param_dict['psf-laser-wavelength'] = np.clip(wavelength, 1e-3, np.inf)

    def set_tile(self, tile):
        if not hasattr(self, 'tile') or (self.tile != tile).any():
            self.tile = tile

    def _kpad(self, field, finalshape, zpad=False, norm=True):
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
        d = np.floor_divide(d, 2)

        if not zpad:
            o[0] = 0

        axes = None
        pad = tuple((d[i]+o[i],d[i]) for i in [0,1,2])
        rpsf = np.pad(field, pad, mode='constant', constant_values=0)
        rpsf = np.fft.ifftshift(rpsf, axes=axes)
        kpsf = fft.rfftn(rpsf, **fftkwargs)

        if norm:
            kpsf /= kpsf[0,0,0]
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
            middle = field.shape[0]//2

            subpsf = self._kpad(self.slices[zslice], fs, norm=True)
            subfield = np.roll(field, middle - i, axis=0)
            subfield = subfield[middle-fs[0]//2:middle+fs[0]//2+1]

            kshape = subfield.shape
            kfield = fft.rfftn(subfield, **fftkwargs)

            outfield[i] = np.real(fft.irfftn(kfield * subpsf, s=kshape, **fftkwargs))[self.support[0]//2]

        return outfield

    def nopickle(self):
        return super(ExactPSF, self).nopickle() + [
            '_rx', '_ry', '_rz', '_rlen',
            '_memoize_clear', '_memoize_caches',
            'rpsf', 'kpsf',
            'cheb', 'slices'
        ]

    def __getstate__(self):
        odict = self.__dict__.copy()
        util.cdd(odict, self.nopickle())
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self.patch({'global_zscale': False})
        if self.shape:
            self.initialize()

class ExactLineScanConfocalPSF(ExactPSF):
    def __init__(self, shape=None, zrange=None, laser_wavelength=0.488, zslab=0.,
            zscale=1.0, kfki=0.889, n2n1=1.44/1.518, alpha=1.173, polar_angle=0.,
            pxsize=0.125, support_factor=2, normalize=False, sigkf=0.0,
            nkpts=None, cutoffval=None, measurement_iterations=None,
            k_dist='gaussian', use_J1=True, sph6_ab=None, global_zscale=False,
            cutbyval=False, cutfallrate=0.25, cutedgeval=1e-12,
            pinhole_width=None, do_pinhole=False, *args, **kwargs):
        """
        PSF for line-scanning confocal microscopes that can be used with the
        peri framework.  Calculates the spatially varying point spread
        function for confocal microscopes and allows them to be applied to
        images as a convolution.

        This PSF assumes that the z extent is large compared to the image size
        and so calculates the local PSF for every z layer in the image.

        Parameters
        ----------
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

        cutbyval : boolean
            If True, cuts the PSF based on the actual value instead of the
            position associated with the nearest value.

        cutfallrate : float
            The relative value of the cutoffval over which to damp the
            remaining values of the psf. 0.3 seems a good fit now.

        cutedgeval : float
            The value with which to determine the edge of the psf, typically
            taken around floating point, 1e-12

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
        super(ExactLineScanConfocalPSF, self).__init__(*args, shape=shape,
                zrange=zrange, laser_wavelength=laser_wavelength, zslab=zslab,
                zscale=zscale, kfki=kfki, n2n1=n2n1, alpha=alpha, pxsize=
                pxsize, polar_angle=polar_angle, support_factor=support_factor,
                normalize=normalize, sigkf=sigkf, nkpts=nkpts, cutoffval=
                cutoffval, measurement_iterations= measurement_iterations,
                k_dist=k_dist, use_J1=use_J1, sph6_ab=sph6_ab, global_zscale=
                global_zscale, cutbyval=cutbyval, cutfallrate=cutfallrate,
                cutedgeval=cutedgeval, **kwargs)

        self.do_pinhole = do_pinhole
        if (pinhole_width is not None) or do_pinhole:
            self.num_line_pts = 3
        else:
            self.num_line_pts = 1
        pinhole_width = pinhole_width if (pinhole_width is not None) else 1.0

        # the next statements must occur in the correct order so that
        # other parameters are not deleted by mistake
        if self.do_pinhole:
            k = 'psf-pinhole-width'
            params.append(k)
            #FIXME this is not correct!!!! should be done intelligently thru
            #the superclass's init, but..
            self.param_dict.update({k:pinhole_width})

    def pack_args(self):
        """
        Pack the parameters into the form necessary for the integration
        routines above.  For example, packs for calculate_linescan_psf
        """
        mapper = {
            'psf-kfki': 'kfki',
            'psf-alpha': 'alpha',
            'psf-n2n1': 'n2n1',
            'psf-sigkf': 'sigkf',
            'psf-sph6-ab': 'sph6_ab',
            'psf-laser-wavelength': 'laser_wavelength',
            'psf-pinhole-width': 'pinhole_width'
        }
        bads = [self.zscale, 'psf-zslab']

        d = {}
        for k,v in iteritems(mapper):
            if k in self.param_dict:
                d[v] = self.param_dict[k]

        d.update({
            'polar_angle': self.polar_angle,
            'normalize': self.normalize,
            'include_K3_det':self.use_J1
        })

        if self.polychromatic:
            d.update({'nkpts': self.nkpts})
            d.update({'k_dist': self.k_dist})

        if self.do_pinhole:
            d.update({'nlpts': self.num_line_pts})

        d.update({'use_laggauss': True})
        return d

    def psffunc(self, *args, **kwargs):
        """Calculates a linescan psf"""
        if self.polychromatic:
            func = psfcalc.calculate_polychrome_linescan_psf
        else:
            func = psfcalc.calculate_linescan_psf
        return func(*args, **kwargs)

class ExactPinholeConfocalPSF(ExactPSF):
    def __init__(self, shape=None, zrange=None, laser_wavelength=0.488, zslab=0.,
            zscale=1.0, kfki=0.889, n2n1=1.44/1.518, alpha=1.173, polar_angle=0.,
            pxsize=0.125, support_factor=2, normalize=False, sigkf=0.0,
            nkpts=None, cutoffval=None, measurement_iterations=None,
            k_dist='gaussian', use_J1=True, sph6_ab=None, global_zscale=False,
            cutbyval=False, cutfallrate=0.25, cutedgeval=1e-12,
            *args, **kwargs):
        """
        PSF for a pinhole confocal microscopes that can be used with the
        peri framework.  Calculates the spatially varying point spread
        function for confocal microscopes and allows them to be applied to
        images as a convolution.

        This PSF assumes that the z extent is large compared to the image size
        and so calculates the local PSF for every z layer in the image.

        Parameters
        ----------
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

        cutbyval : boolean
            If True, cuts the PSF based on the actual value instead of the
            position associated with the nearest value.

        cutfallrate : float
            The relative value of the cutoffval over which to damp the
            remaining values of the psf. 0.3 seems a good fit now.

        cutedgeval : float
            The value with which to determine the edge of the psf, typically
            taken around floating point, 1e-12

        Notes:
            a = ExactLineScanConfocalPSF((64,)*3)
            psf, (z,y,x) = a.psf_slice(1., size=51)
            imshow((psf*r**4)[:,:,25], cmap='bone')
        """
        super(ExactPinholeConfocalPSF, self).__init__(shape=shape,
                zrange=zrange, laser_wavelength=laser_wavelength, zslab=zslab,
                zscale=zscale, kfki=kfki, n2n1=n2n1, alpha=alpha, pxsize=
                pxsize, polar_angle=polar_angle, support_factor=support_factor,
                normalize=normalize, sigkf=sigkf, nkpts=nkpts, cutoffval=
                cutoffval, measurement_iterations= measurement_iterations,
                k_dist=k_dist, use_J1=use_J1, sph6_ab=sph6_ab, global_zscale=
                global_zscale, cutbyval=cutbyval, cutfallrate=cutfallrate,
                cutedgeval=cutedgeval, *args, **kwargs)

    def pack_args(self):
        """
        Pack the parameters into the form necessary for the integration
        routines above.  For example, packs for calculate_linescan_psf
        """
        mapper = {
            'psf-kfki': 'kfki',
            'psf-alpha': 'alpha',
            'psf-n2n1': 'n2n1',
            'psf-sigkf': 'sigkf',
            'psf-sph6-ab': 'sph6_ab',
            'psf-laser-wavelength': 'laser_wavelength'
        }
        bads = [self.zscale, 'psf-zslab']

        d = {}
        for k,v in iteritems(mapper):
            if k in self.param_dict:
                d[v] = self.param_dict[k]

        d.update({
            'polar_angle': self.polar_angle,
            'normalize': self.normalize,
            'include_K3_det':self.use_J1
        })

        if self.polychromatic:
            d.update({'nkpts': self.nkpts})
            d.update({'k_dist': self.k_dist})

        if self.do_pinhole:
            d.update({'nlpts': self.num_line_pts})
        return d

    def psffunc(self, x, y, z, **kwargs):
        """Calculates a pinhole psf"""
        #do_pinhole?? FIXME
        if self.polychromatic:
            func = psfcalc.calculate_polychrome_pinhole_psf
        else:
            func = psfcalc.calculate_pinhole_psf
        x0, y0 = [psfcalc.vec_to_halfvec(v) for v in [x,y]]
        vls = psfcalc.wrap_and_calc_psf(x0, y0, z, func, **kwargs)
        return vls / vls.sum()

class ChebyshevPSF(ExactPSF):
    def __init__(self, cheb_degree=6, cheb_evals=8, *args, **kwargs):
        """
        Same as ExactPSF, except that the convolution is performed in
        the 4th dimension by employing fast Chebyshev approximates to
        how the PSF varies with depth into the sample. For help, see
        ExactPSF.

        Other parameters
        ----------------
        cheb_degree : integer
            degree of the Chebyshev approximant

        cheb_evals : integer
            number of interpolation points used to create the coefficient matrix

        See also
        --------
        :class:`peri.comp.exactpsf.ExactPSF`
        """
        self.cheb_degree = cheb_degree
        self.cheb_evals = cheb_evals

        super(ChebyshevPSF, self).__init__(*args, **kwargs)

    def update(self, params, values):
        self.update_values(params, values)
        self.characterize_psf()

        self.cheb = interpolation.ChebyshevInterpolation1D(self.psf, window=self.zrange,
                        degree=self.cheb_degree, evalpts=self.cheb_evals)
        return True

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

        kshape = field.shape
        kfield = fft.rfftn(field, **fftkwargs)
        for k,c in enumerate(self.cheb.coefficients):
            pad = self._kpad(c, finalshape=self.tile.shape, zpad=True, norm=False)
            cov = np.real(fft.irfftn(kfield * pad, s=kshape, **fftkwargs))

            outfield += self.cheb.tk(k, zc)[:,None,None] * cov

        return outfield

    def __str__(self):
        return "{} {}".format(self.__class__.__name__, [self.cheb_degree,
                self.cheb_evals])

class FixedSSChebPSF(ChebyshevPSF):
    def __init__(self, support_size=[35,17,25], *args, **kwargs):
        """
        ChebyshevPSF with a fixed support size

        See also
        --------
        :class:`peri.comp.exactpsf.ChebyshevPSF`
        """
        self.cutoffval = None
        self.support = np.array(support_size)
        super(FixedSSChebPSF, self).__init__(*args, **kwargs)

    def characterize_psf(self):
        """ Get support size and drift polynomial for current set of params """
        l,u = max(self.zrange[0], self.param_dict['psf-zslab']), self.zrange[1]

        size_l, drift_l = self.measure_size_drift(l, size=self.support)
        size_u, drift_u = self.measure_size_drift(u, size=self.support)

        self.drift_poly = np.polyfit([l, u], [drift_l, drift_u], 1)

    def __str__(self):
        return "{} {}".format(self.__class__.__name__, self.support)

##Multiple-inheritance-defined classes, see individual docs for details
class ChebyshevLineScanConfocalPSF(ChebyshevPSF, ExactLineScanConfocalPSF):
    """See ChebyshevPSF, ExactLineScanConfocalPSF for docs."""
    pass

class FixedSSChebLinePSF(FixedSSChebPSF, ExactLineScanConfocalPSF):
    """See FixedSSChebPSF, ExactLineScanConfocalPSF for docs."""
    pass

class ChebyshevPinholeConfocalPSF(ChebyshevPSF, ExactPinholeConfocalPSF):
    """See ChebyshevPSF, ExactPinholeConfocalPSF for docs."""
    pass

class FixedSSChebPinholePSF(FixedSSChebPSF, ExactPinholeConfocalPSF):
    """See FixedSSChebPSF, ExactPinholeConfocalPSF for docs."""
    pass
