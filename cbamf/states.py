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

def save_wisdom():
    pickle.dump(pyfftw.export_wisdom(), open(WISDOM_FILE, 'w'))

try:
    with open(WISDOM_FILE) as wisdom:
        pyfftw.import_wisdom(pickle.load(open(WISDOM_FILE)))
except IOError as e:
    save_wisdom()

(
    PSF_NONE,
    PSF_ISOTROPIC_DISC,
    PSF_ANISOTROPIC_GAUSSIAN,
    PSF_ISOTROPIC_PADE_3_7
) = xrange(4)

FFTW_PLAN_FAST = 'FFTW_ESTIMATE'
FFTW_PLAN_NORMAL = 'FFTW_MEASURE'
FFTW_PLAN_SLOW = 'FFTW_PATIENT'

psf_nparams = {
    PSF_NONE: 0,
    PSF_ISOTROPIC_DISC: 2,
    PSF_ANISOTROPIC_GAUSSIAN: 2,
    PSF_ISOTROPIC_PADE_3_7: 12
}

def loadtest():
    import pickle
    itrue, xstart, rstart, pstart, ipure = pickle.load(open("/media/scratch/bamf/bamf_ic_16.pkl", 'r'))
    bkg = np.zeros((3,3,3))
    bkg[0,0,0] = 1
    state = np.hstack([xstart.flatten(), rstart, pstart, bkg.ravel(), np.ones(1), np.zeros(1)])
    return ConfocalImagePython(len(rstart), itrue, pad=32, order=(3,3,3), state=state, fftw_planning_level=FFTW_PLAN_SLOW)

class State(object):
    def __init__(self, nparams, state=None):
        self.nparams = nparams
        self.state = state if state is not None else np.zeros(self.nparams, dtype='float32')
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
    def __init__(self, N, image, psftype=PSF_ANISOTROPIC_GAUSSIAN, pad=16, order=1, sigma=0.1,
            fftw_planning_level=FFTW_PLAN_NORMAL, threads=-1, *args, **kwargs):
        self.N = N
        self.image = image
        self.psftype = psftype
        self.psfn = psf_nparams[self.psftype]
        self.pad = pad
        self.index = None
        self.sigma = sigma

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

        self.initialize()

    def _psf_disc(self):
        params = self.state[self.b_psf]
        return (1.0 + np.exp(-params[1]*params[0])) / (1.0 + np.exp(params[1]*(self._klen - params[0])))

    def _psf_gaussian_rz(self):
        params = self.state[self.b_psf]
        return np.exp(-(self._kx*params[0])**2 - (self._ky*params[0])**2 - (self._kz*params[1])**2)

    def _setup_ffts(self):
        self._fftn_data = pyfftw.n_byte_align_empty(self._shape_fft, 16, dtype='complex')
        self._ifftn_data = pyfftw.n_byte_align_empty(self._shape_fft, 16, dtype='complex')
        self._fftn = fftn(self._fftn_data, overwrite_input=True,
                planner_effort=self.fftw_planning_level, threads=self.threads)
        self._ifftn = ifftn(self._ifftn_data, overwrite_input=True,
                planner_effort=self.fftw_planning_level, threads=self.threads)

    def _setup_kvecs(self):
        kz = 2*np.pi*np.fft.fftfreq(self._shape_fft[0])[:,None,None]
        ky = 2*np.pi*np.fft.fftfreq(self._shape_fft[1])[None,:,None]
        kx = 2*np.pi*np.fft.fftfreq(self._shape_fft[2])[None,None,:]
        self._kx, self._ky, self._kz = kx, ky, kz
        self._kvecs = np.rollaxis(np.array(np.broadcast_arrays(kz,ky,kx)), 0, 4)
        self._klen = np.sqrt(kx**2 + ky**2 + kz**2)

    def _setup_rvecs(self):
        z,y,x = np.meshgrid(*(xrange(i) for i in self.image.shape), indexing='ij')
        self._rvecs = np.rollaxis(np.array(np.broadcast_arrays(z,y,x)), 0, 4)

    def _rparticle(self, pos, rad, field, sign=1):
        p = np.round(pos)
        r = np.ceil(rad)+1

        sl = np.s_[p[0]-r:p[0]+r, p[1]-r:p[1]+r, p[2]-r:p[2]+r]
        subr = self._rvecs[sl + (np.s_[:],)]
        rvec = (subr - pos)

        # apply the z-scaling of pixels
        zscale = self.state[self.b_zscale]
        rvec[...,0] *= zscale

        rdist = np.sqrt((rvec**2).sum(axis=-1))
        field[sl] += sign/(1.0 + np.exp(5*(rdist - rad)))

    def initialize(self):
        self.create_base_platonic_image()
        self.create_bkg_field()

    def update_rspace_spheres(self, pos0, rad0, pos1, rad1):
        self._rparticle(pos0, rad0, self.field_particles, -1)
        self._rparticle(pos1, rad1, self.field_particles, +1)

    def create_base_platonic_image(self):
        self._setup_rvecs()
        self.field_particles = np.zeros(self.image.shape)

        for p0, r0 in zip(self.state[self.b_pos].reshape(-1,3), self.state[self.b_rad]):
            self._rparticle(p0, r0, self.field_particles)

    def create_bkg_field(self):
        self.field_bkg = self.poly.evaluate(self.state[self.b_bkg])

    def create_kpsf(self):
        if self.psftype == PSF_ISOTROPIC_DISC:
            self._kpsf = self._psf_disc()
        if self.psftype == PSF_ANISOTROPIC_GAUSSIAN:
            self._kpsf = self._psf_gaussian_rz()

    def create_final_image(self):
        if (not pyfftw.is_n_byte_aligned(self._fftn_data, 16) or
            not pyfftw.is_n_byte_aligned(self._ifftn_data, 16)):
            raise AttributeError("FFT arrays became misaligned")

        self._fftn_data[:] = self.field_bkg[self._slice] * (1 - self.field_particles[self._slice])
        self._fftn.execute()
        self._ifftn_data[:] = self._fftn.get_output_array() * self._kpsf / self._fftn_data.size
        self._ifftn.execute()

        self.model_image = np.real(self._ifftn.get_output_array()) + self.state[self.b_amp]
        return self.model_image

    def create_differences(self):
        return (
            self.model_image[self._cmp_region][self._cmp_mask]-
            self.image[self._cmp_slice][self._cmp_mask]
        )

    def set_current_particle(self, index=None, sub_image_size=None):
        """
        We must set up the following structure:
        +-----------------------------+
        |        Buffer Region        |
        |(bkg field + other particles)|
        |                             |
        |    +-------------------+    |
        |    |                   |    |
        |    |    Comparison     |    |
        |    |      Region       |    |
        |    |                   |    |
        |    |                   |    |
        |    |      (size)       |    |
        |    +-------------------+    |
        |                             |
        |           (pad)             |
        +-----------------------------+
        """
        pos = self.state[self.b_pos].reshape(-1,3)
        rad = self.state[self.b_rad]

        if index is not None:
            self.index = index

            size = sub_image_size or int(3*rad[index])
            center = np.round(pos[index]).astype('int32')
            pl = (center - size/2 - self.pad/2).astype('int')
            pr = (center + size/2 + self.pad/2).astype('int')
        else:
            self.index = -1

            pl, pr = np.array([0,0,0]), np.array(self.image.shape)
            center = (pr - pl)/2

        if (pl < 0).any() or (pr > self.image.shape).any():
            return False

        lcmp = np.vstack([pl, np.zeros(pl.shape)])
        rcmp = np.vstack([pr, self.image.shape])
        pl = np.max(lcmp, axis=0).astype('int')
        pr = np.min(rcmp, axis=0).astype('int')

        # these variables map the buffer region back
        # into the large image in real space
        self._center = center
        self._bounds = (pl, pr)
        self._slice = np.s_[pl[0]:pr[0], pl[1]:pr[1], pl[2]:pr[2]]
        self._shape_fft = np.abs(pr - pl)

        # these variables have to do with the comparison region
        # that is inside the buffer region
        inl, inr = pl + self.pad/2, pr - self.pad/2
        self._cmp_region = (np.s_[self.pad/2:-self.pad/2],)*3
        self._cmp_slice = np.s_[inl[0]:inr[0], inl[1]:inr[1], inl[2]:inr[2]]
        self._cmp_mask = self.image[self._cmp_slice] > -10

        self._setup_kvecs()
        self._setup_ffts()
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
            self.update_rspace_spheres(pos0, rad0, pos1, rad1)

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

    def _grad_single_param(self, block, dl):
        self.push_change(block, self.state[block]+dl)
        self.create_final_image()
        loglr = self.loglikelihood()
        self.pop_change()

        self.push_change(block, self.state[block]-dl)
        self.create_final_image()
        logll = self.loglikelihood()
        self.pop_change()

        return (loglr - logll) / (2*dl)

    def gradloglikelihood(self, dl=1e-3):
        grad = []
        for pg in self.param_order:
            print '{:-^39}'.format(' '+pg.upper()+' ')
            if pg == 'pos':
                for i in xrange(self.N):
                    self.set_current_particle(i)
                    blocks = self.blocks_particle()[:-1]

                    for block in blocks:
                        grad.append(self._grad_single_param(block, dl))

            if pg == 'rad':
                for i in xrange(self.N):
                    self.set_current_particle(i)
                    block = self.blocks_particle()[-1]
                    grad.append(self._grad_single_param(block, dl))

            if pg == 'psf' or pg == 'bkg' or pg == 'amp' or pg == 'zscale':
                self.set_current_particle()
                blocks = self.explode(self.create_block(pg))

                for block in blocks:
                    grad.append(self._grad_single_param(block, dl))

        return np.array(grad)

    def loglikelihood(self):
        self.create_final_image()
        return -(self.create_differences()**2).sum() / self.sigma**2

    def __del__(self):
        save_wisdom()
