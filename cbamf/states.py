import os
import pickle
import numpy as np
from itertools import product
from collections import OrderedDict
from copy import deepcopy

import pyfftw
from pyfftw.builders import fftn, ifftn
from multiprocessing import cpu_count

WISDOM_FILE = os.path.join(os.path.expanduser("~"), ".fftw_wisdom.pkl")
pyfftw.import_wisdom(pickle.load(open(WISDOM_FILE)))

(
    PSF_NONE,
    PSF_ISOTROPIC_DISC,
    PSF_ISOTROPIC_GAUSSIAN,
    PSF_ISOTROPIC_PADE_3_7
) = xrange(4)

FFTW_PLAN_FAST = 'FFTW_ESTIMATE'
FFTW_PLAN_NORMAL = 'FFTW_MEASURE'
FFTW_PLAN_SLOW = 'FFTW_PATIENT'

psf_nparams = {
    PSF_NONE: 0,
    PSF_ISOTROPIC_DISC: 2,
    PSF_ISOTROPIC_GAUSSIAN: 2,
    PSF_ISOTROPIC_PADE_3_7: 12
}

def loadtest():
    import pickle
    itrue, xstart, rstart, pstart, ipure = pickle.load(open("/media/scratch/bamf/bamf_ic_16.pkl", 'r'))
    bkg = np.zeros((3,3,3))
    bkg[0,0,0] = 1
    state = np.hstack([xstart.flatten(), rstart, pstart, bkg.ravel(), np.ones(1.0)])
    return ConfocalImagePython(len(rstart), itrue, pad=32, order=(3,3,3), state=state, fftw_planning_level=FFTW_PLAN_SLOW)

class State(object):
    def __init__(self, nparams, state=None):
        self.nparams = nparams
        self.state = state if state is not None else np.zeros(self.nparams, dtype='float32')
        self.loglikelihood = None
        self.gradloglikelihood = None
        self.stack = []

    def push_change(self, block, data):
        curr = self.state[block].copy()
        self.stack.append((block, curr))
        self.update(block, data)

    def pop_change(self):
        block, data = self.stack.pop()
        self.update(block, data)

    def update(self, block, data):
        return self._update_state(block, data)

    def _update_state(self, block, data):
        self.state[block] = data.astype(self.state.dtype)

    def block_all(self):
        return np.ones(self.nparams, dtype='bool')

    def block_none(self):
        return np.zeros(self.nparams, dtype='bool')

    def block_range(self, bmin, bmax):
        block = self.block_none()
        bmin = max(bmin, 0)
        bmax = min(bmax, self.nparams)
        block[bmin:bmax] = True
        return block

    def blocks_const_size(self, size):
        blocks = []
        end = self.nparams/size if (self.nparams % size == 0) else self.nparams / size + 1
        for i in xrange(end):
            ma = self.block_none()

            if (i+1)*size > self.nparams:
                ma[i*size:] = True
            else:
                ma[i*size:(i+1)*size] = True

            blocks.append(ma)
        return blocks

    def set_state(self, state):
        self.state = state.astype(self.state.dtype)

    def reset(self):
        self.state *= 0

class PolyField3D(object):
    def __init__(self, shape, order=(1,1,1)):
        self.shape = shape
        self.order = order
        self.setup_rvecs()

    def poly_orders(self):
        return product(*(xrange(o) for o in self.order))

    def setup_rvecs(self):
        o = self.shape
        self.rx, self.ry, self.rz = np.mgrid[0:o[0], 0:o[1], 0:o[2]] / float(max(o))
        self.poly = []

        for i,j,k in self.poly_orders():
            self.poly.append( self.rx**i * self.ry**j * self.rz**k )

        self.poly = np.rollaxis( np.array(self.poly), 0, len(self.shape)+1 )

    def evaluate(self, coeffs, sl=np.s_[:,:,:]):
        if len(sl) != 3:
            raise AttributeError("Slice object must be 3D as well")
        sl = sl + (np.s_[:],)
        return (self.poly[sl] * coeffs).sum(axis=-1)

class ConfocalImagePython(State):
    def __init__(self, N, image, psftype=PSF_ISOTROPIC_DISC, pad=16, order=1,
            fftw_planning_level=FFTW_PLAN_NORMAL, threads=-1, *args, **kwargs):
        self.N = N
        self.image = image
        self.image_mask = self.image > -10
        self.psftype = psftype
        self.psfn = psf_nparams[self.psftype]
        self.pad = pad
        self.field_platonic = None
        self.field_bkg = None
        self.index = None

        self.threads = threads if threads > 0 else cpu_count()
        self.fftw_planning_level = fftw_planning_level
        self.order = order if hasattr(order, "__iter__") else (order,)*3
        self.poly = PolyField3D(shape=self.image.shape, order=self.order)

        self.param_dict = OrderedDict({
            'pos': 3*self.N,
            'rad': self.N,
            'typ': 0,
            'psf': self.psfn,
            'bkg': np.prod(self.order),
            'amp': 1,
            'zscale': 1
        })

        self.param_order = ['pos', 'rad', 'typ', 'psf', 'bkg', 'amp', 'zscale']
        self.param_lengths = [self.param_dict[k] for k in self.param_order]

        total_params = sum(self.param_lengths)
        super(ConfocalImagePython, self).__init__(nparams=total_params, *args, **kwargs)

        self.b_pos = self.create_block('pos')
        self.b_rad = self.create_block('rad')
        self.b_psf = self.create_block('psf')
        self.b_bkg = self.create_block('bkg')
        self.b_amp = self.create_block('amp')
        self.b_zscale = self.create_block('zscale')

    def _disc1(self, k, R):
        return 2*R*np.sin(k)/k

    def _disc2(self, k, R):
        return 2*np.pi*R**2 * j1(k) / k

    def _disc3(self, k, R):
        return 4*np.pi*R**3 * (np.sin(k)/k - np.cos(k))/k**2

    def _psf_disc(self):
        params = self.state[self.b_psf]
        return (1.0 + np.exp(-params[1]*params[0])) / (1.0 + np.exp(params[1]*(self._klen - params[0])))

    def _psf_gaussian_rz(self):
        params = self.state[self.b_psf]
        return np.exp(-(self._kx**2 + self._ky**2)*params[0]**2/2 - self._kz**2*params[1]**2/2)

    def _setup_ffts(self):
        self._fftn_data = pyfftw.n_byte_align_empty(self._shape_fft, 16, dtype='complex')
        self._ifftn_data = pyfftw.n_byte_align_empty(self._shape_fft, 16, dtype='complex')
        self._fftn = fftn(self._fftn_data, overwrite_input=True,
                planner_effort=self.fftw_planning_level, threads=self.threads)
        self._ifftn = ifftn(self._ifftn_data, overwrite_input=True,
                planner_effort=self.fftw_planning_level, threads=self.threads)

    def _setup_kvecs(self):
        zscale = self.state[self.b_zscale]
        kz = 2*np.pi*np.fft.fftfreq(self._shape_fft[0])[:,None,None]*zscale
        ky = 2*np.pi*np.fft.fftfreq(self._shape_fft[1])[None,:,None]
        kx = 2*np.pi*np.fft.fftfreq(self._shape_fft[2])[None,None,:]
        self._kx, self._ky, self._kz = kx, ky, kz
        self._kvecs = np.rollaxis(np.array(np.broadcast_arrays(kz,ky,kx)), 0, 4)
        self._klen = np.sqrt(kx**2 + ky**2 + kz**2)

    def _setup_rvecs(self):
        z,y,x = np.meshgrid(*(xrange(i) for i in self._shape_fft), indexing='ij')
        self._rvecs = np.rollaxis(np.array(np.broadcast_arrays(z,y,x)), 0, 4)

    def _kparticle(self, pos, rad):
        kdotx = (pos * self._kvecs).sum(axis=-1)
        return self._disc3(self._klen*rad+1e-8, rad)*np.exp(-1.j*kdotx)

    def _rparticle(self, pos, rad, field, sign=1):
        p = np.round(pos)
        r = np.ceil(rad)+1

        sl = np.s_[p[0]-r:p[0]+r, p[1]-r:p[1]+r, p[2]-r:p[2]+r]
        subr = self._rvecs[sl + (np.s_[:],)]
        rvec = (subr - pos)

        # apply the z-scaling of pixels
        zscale = self.state[self.b_zscale]
        rvec[...,0] /= zscale

        rdist = np.sqrt((rvec**2).sum(axis=-1))
        field[sl] += sign/(1.0 + np.exp(5*(rdist - rad)))

    def update_kspace_spheres(self, pos0, rad0, pos1, rad1):
        self.field_particles -= self._kparticle(pos0, rad0)
        self.field_particles += self._kparticle(pos1, rad1)

    def update_rspace_spheres(self, pos0, rad0, pos1, rad1):
        self._rparticle(pos0, rad0, self.field_particles, -1)
        self._rparticle(pos1, rad1, self.field_particles, +1)

    def create_base_platonic_image_kspace(self):
        if self.index is None:
            raise AttributeError("Particle index has not been selected, call set_current_particle")

        self.field_particles = np.zeros(self._shape_fft, dtype='complex')

        for p0, r0 in zip(self._pos.reshape(-1,3), self._rad):
            pos = p0 - self._bounds[0]
            self.field_particles += self._kparticle(pos, r0)

    def create_base_platonic_image(self):
        if self.index is None:
            raise AttributeError("Particle index has not been selected, call set_current_particle")

        self._setup_rvecs()
        self.field_particles = np.zeros(self._shape_fft)

        for p0, r0 in zip(self._pos.reshape(-1,3), self._rad):
            pos = p0 - self._bounds[0]
            self._rparticle(pos, r0, self.field_particles)

    def create_bkg_field(self):
        self.field_bkg = self.poly.evaluate(self.state[self.b_bkg], self._slice)

    def create_kpsf(self):
        if self.psftype == PSF_ISOTROPIC_DISC:
            self._kpsf = self._psf_disc()
        if self.psftype == PSF_ISOTROPIC_GAUSSIAN:
            self._kpsf = self._psf_gaussian_rz()

    def create_final_image(self):
        if (not pyfftw.is_n_byte_aligned(self._fftn_data, 16) or
            not pyfftw.is_n_byte_aligned(self._ifftn_data, 16)):
            raise AttributeError("FFT arrays became misaligned")

        self._fftn_data[:] = self.field_bkg * (1 - self.field_particles)
        self._fftn.execute()
        self._ifftn_data[:] = self._fftn.get_output_array() * self._kpsf / self._fftn_data.size
        self._ifftn.execute()

        self.model_image = np.real(self._ifftn.get_output_array()) + self.state[self.b_amp]
        return self.model_image

    def create_differences(self):
        return (
            self.model_image[self._cmp2buffer][self._cmp_mask]-
            self.image[self._cmp_slice][self._cmp_mask]
        )

    def set_current_particle(self, index=None, sub_image_size=None):
        """
        We must set up the following structure:
        +-----------------------------+-----+
        |        Buffer Region        |     |
        |(bkg field + other particles)|     |
        |                             |     |
        |    +-------------------+    |     |
        |    |                   |    |     |
        |    |    Comparison     |    |     |
        |    |      Region       |    |     |
        |    |                   |    |     |
        |    |                   |    |     |
        |    |      (size)       |    |     |
        |    +-------------------+    |     |
        |                             |     |
        |           (pad)             |     |
        +-----------------------------+     |
        |                                   |
        |       FFT padding region          |
        +-----------------------------------+
        """
        pos = self.state[self.b_pos].reshape(-1,3)
        rad = self.state[self.b_rad]

        if index is not None:
            self.index = index

            size = sub_image_size or self.pad
            center = np.round(pos[index]).astype('int32')
            pl = (center - size/2 - self.pad/2).astype('int')
            pr = (center + size/2 + self.pad/2).astype('int')
        else:
            self.index = -1

            pl = np.array([0,0,0])
            pr = np.array(self.image.shape)
            center = (pr - pl)/2

        if (pl < 0).any() or (pr > self.image.shape).any():
            return False

        # these variables map the buffer region back
        # into the large image in real space
        self._mask = ((pos > pl+rad[:,None]) & (pos < pr-rad[:,None])).all(axis=-1)
        self._center = center
        self._bounds = (pl, pr)
        self._slice = np.s_[pl[0]:pr[0], pl[1]:pr[1], pl[2]:pr[2]]
        self._shape = np.abs(pr - pl)
        self._shape_fft = self._shape

        self._pos = pos[self._mask].flatten()
        self._rad = rad[self._mask]

        # these variables have to do with the comparison region
        # that is inside the buffer region
        inl = pl + self.pad/2
        inr = pr - self.pad/2
        self._cmp_bounds = (inl, inr)
        self._cmp2buffer = (np.s_[self.pad/2:-self.pad/2],)*3
        self._cmp_slice = np.s_[inl[0]:inr[0], inl[1]:inr[1], inl[2]:inr[2]]
        self._cmp_shape = np.abs(inr - inl)
        self._cmp_mask = self.image[self._cmp_slice] > -10

        self._setup_kvecs()
        self._setup_ffts()
        self.create_base_platonic_image()
        self.create_bkg_field()
        self.create_kpsf()
        return True

    def update(self, block, data):
        pmask = block[self.b_pos]
        rmask = block[self.b_rad]
        bmask = block[self.b_bkg]

        pmask = pmask.reshape(-1,3)
        particles = pmask.any(axis=-1) | rmask

        pos0 = self.state[self.b_pos].copy().reshape(-1,3)[particles].flatten()
        rad0 = self.state[self.b_rad].copy()[particles]

        self._update_state(block, data)

        pos1 = self.state[self.b_pos].copy().reshape(-1,3)[particles].flatten()
        rad1 = self.state[self.b_rad].copy()[particles]

        if len(pos1) > 0 and len(rad1) > 0:
            cpos0 = (pos0.reshape(-1,3) - self._bounds[0]).flatten()
            cpos1 = (pos1.reshape(-1,3) - self._bounds[0]).flatten()
            self.update_rspace_spheres(cpos0, rad0, cpos1, rad1)

        # if the psf was changed, update
        if block[self.b_psf].any():
            self.create_kpsf()

        # update the background if it has been changed
        if block[self.b_bkg].any():
            self.create_bkg_field()

        # we actually don't have to do anything if the amplitude is changed
        if block[self.b_amp].any():
            pass

        if block[self.b_zscale].any():
            self._setup_kvecs()
            self._setup_ffts()
            self.create_base_platonic_image()
            self.create_bkg_field()
            self.create_kpsf()

    def create_clips(self, ccd_size):
        clip_pos = np.array([
            [0, ccd_size[0]],
            [0, ccd_size[1]],
            [0, ccd_size[2]]
        ])
        clip_rad = np.array([1,50])
        clip_psf = np.array([0, 100])
        clips = np.vstack([clip_pos]*self.N + [clip_rad]*self.N + [clip_psf]*self.psfn)
        return clips

    def blocks_particles(self):
        masks = []
        for i in xrange(self.N):
            mask = self.block_none()
            mask[i*3:(i+1)*3] = True
            mask[3*self.N+i] = True
            masks.append(mask)
        return masks

    def blocks_particle(self):
        if self.index is None:
            raise AttributeError("No particle selected, run set_current_particle")

        p_ind, r_ind = 3*self.index, 3*self.N + self.index

        blocks = []
        for t in xrange(p_ind, p_ind+3):
            blocks.append(self.block_range(t, t+1))
        blocks.append(self.block_range(r_ind, r_ind+1))
        return blocks

    def create_block(self, typ='all'):
        return self.block_range(*self._block_offset_end(typ))

    def explode(self, block):
        inds = np.arange(block.shape[0])
        inds = inds[block]

        blocks = []
        for i in inds:
            tblock = self.block_none()
            tblock[i] = True
            blocks.append(tblock)
        return blocks

    def _block_offset_end(self, typ='pos'):
        index = self.param_order.index(typ)
        off = sum(self.param_lengths[:index])
        end = off + self.param_lengths[index]
        return off, end

    def grad(self, dl=1e-3):
        blocks = self.explode(self.block_all())
        grad = []
        for block in blocks:
            self.push_change(block, self.state[block]+dl)
            self.create_final_image()
            loglr = -(self.create_differences()**2).sum()
            self.pop_change()

            self.push_change(block, self.state[block]-dl)
            self.create_final_image()
            logll = -(self.create_differences()**2).sum()
            self.pop_change()

            grad.append((loglr - logll) / (2*dl))
        return np.array(grad)

    def __del__(self):
        pickle.dump(pyfftw.export_wisdom(), open(WISDOM_FILE, 'w'))
