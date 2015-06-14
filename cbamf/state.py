import itertools
import numpy as np
from copy import deepcopy
from cbamf.cu import fields

psf_nparams = {
    fields.PSF_NONE: 0,
    fields.PSF_ISOTROPIC_DISC: 2,
    fields.PSF_ISOTROPIC_GAUSSIAN: 2,
    fields.PSF_ISOTROPIC_PADE_3_7: 12
}

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
        return itertools.product(*(xrange(o) for o in self.order))

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

class StateXRPBA(State):
    """
    This class assumes that the positions and true image are indeed
    padded just as the fake image should be
    """

    def __init__(self, N, image, psftype=fields.PSF_ISOTROPIC_DISC, pad=16, order=1, *args, **kwargs):
        self.N = N
        self.image = image
        self.image_mask = self.image > -10
        self.psftype = psftype
        self.psfn = psf_nparams[self.psftype]
        self.pad = pad
        self.kccd = None
        self.field_kspace_platonic = None
        self.field_bkg = None
        self.index = None
        self.impack = None

        self.order = order if hasattr(order, "__iter__") else (order,)*3
        self.poly = PolyField3D(shape=self.image.shape, order=self.order)

        self.param_order = ['pos', 'rad', 'typ', 'psf', 'bkg', 'amp']
        self.param_lengths = [3*self.N, self.N, 0, self.psfn, np.prod(self.order), 1]

        total_params = sum(self.param_lengths)
        super(StateXRPBA, self).__init__(nparams=total_params, *args, **kwargs)

        self.b_pos = self.create_block('pos')
        self.b_rad = self.create_block('rad')
        self.b_psf = self.create_block('psf')
        self.b_bkg = self.create_block('bkg')
        self.b_amp = self.create_block('amp')

    def create_kspace_platonic_image(self):
        if self.index is None:
            raise AttributeError("Particle index has not been selected, call set_current_particle")

        if self.field_kspace_platonic:
            fields.freeCFieldGPU(self.field_kspace_platonic)

        self.sub_size = np.array([i+self.pad for i in self.sub_shape[::-1]], dtype='int32')
        self.kccd = np.zeros(self.sub_size, dtype='complex64').flatten()

        self.field_kspace_platonic = fields.py_cfloat2cfield(self.kccd, self.sub_size)

        centered_pos = (self.sub_pos.reshape(-1,3) - self.bounds[0]).flatten()
        fields.create_kspace_spheres(self.field_kspace_platonic, centered_pos, self.sub_rad)

    def create_bkg_field(self):
        self.field_bkg = self.poly.evaluate(self.state[self.b_bkg], self.sub_slice)

    def create_final_image(self):
        t = np.zeros(self.sub_shape[::-1], dtype='float32').flatten()
        field = fields.createField(np.array(self.sub_shape[::-1], dtype='int32'))
        fields.fieldSet(field, t)

        psf = self.state[self.b_psf]
        self.impack = self.impack or fields.create_image_package(field, self.pad)

        fields.update_image(self.field_kspace_platonic, field, self.impack, psf.astype('float32'), self.psftype, self.pad)
        fields.fieldGet(field, t)
        fields.freeField(field)

        self.model_image = self.state[self.b_amp]+t.reshape(self.sub_shape)
        return self.model_image

    def set_current_particle(self, index=None, max_size=10):
        pos = self.state[self.b_pos].reshape(-1,3)
        rad = self.state[self.b_rad]
        x,y,z = pos.T

        if index is not None:
            self.index = index

            size = max_size
            center = np.round(pos[index]).astype('int32')
            pl = (center - size/2).astype('int')
            pr = (center + size/2).astype('int')
        else:
            self.index = -1

            pl = np.array([0,0,0])
            pr = np.array(self.image.shape[::-1])
            center = (pr - pl)/2

        if (pl < 0).any() or (pr[::-1] > self.image.shape).any():
            print pl
            print pr
            print self.image.shape
            return False

        mask =  (x > pl[0]-self.pad/2) & (x < pr[0]+self.pad/2)
        mask &= (y > pl[1]-self.pad/2) & (y < pr[1]+self.pad/2)
        mask &= (z > pl[2]-self.pad/2) & (z < pr[2]+self.pad/2)

        self.mask = mask
        self.center = center
        self.bounds = (pl, pr)

        self.sub_pos = pos[mask].flatten()
        self.sub_rad = rad[mask]

        self.sub_slice = np.s_[pl[2]:pr[2], pl[1]:pr[1], pl[0]:pr[0]]
        self.sub_shape = np.abs(pr - pl)[::-1]

        self.sub_im_compare = self.image[self.sub_slice] > -10

        if self.impack is not None:
            fields.free_image_package(self.impack)
            self.impack = None

        self.create_kspace_platonic_image()
        self.create_bkg_field()
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
            cpos0 = (pos0.reshape(-1,3) - self.bounds[0]).flatten()
            cpos1 = (pos1.reshape(-1,3) - self.bounds[0]).flatten()
            fields.update_kspace_spheres(self.field_kspace_platonic, cpos0, rad0, cpos1, rad1)

        if bmask.any():
            self.create_bkg_field()

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


def loadtest():
    import pickle
    itrue, xstart, rstart, pstart, ipure = pickle.load(open("/media/scratch/bamf/bamf_ic_16.pkl", 'r'))
    bkg = np.zeros((3,3,3))
    bkg[0,0,0] = 1
    state = np.hstack([xstart.flatten(), rstart, pstart, bkg.ravel(), np.ones(1.0)])
    return ConfocalImagePython(len(rstart), itrue, pad=32, order=(3,3,3), state=state)

class ConfocalImagePython(State):
    def __init__(self, N, image, psftype=fields.PSF_ISOTROPIC_DISC, pad=16, order=1, *args, **kwargs):
        self.N = N
        self.image = image
        self.image_mask = self.image > -10
        self.psftype = psftype
        self.psfn = psf_nparams[self.psftype]
        self.pad = pad
        self.field_platonic = None
        self.field_bkg = None
        self.index = None

        self.order = order if hasattr(order, "__iter__") else (order,)*3
        self.poly = PolyField3D(shape=self.image.shape, order=self.order)

        self.param_order = ['pos', 'rad', 'typ', 'psf', 'bkg', 'amp']
        self.param_lengths = [3*self.N, self.N, 0, self.psfn, np.prod(self.order), 1]

        total_params = sum(self.param_lengths)
        super(ConfocalImagePython, self).__init__(nparams=total_params, *args, **kwargs)

        self.b_pos = self.create_block('pos')
        self.b_rad = self.create_block('rad')
        self.b_psf = self.create_block('psf')
        self.b_bkg = self.create_block('bkg')
        self.b_amp = self.create_block('amp')

    def _disc1(self, k, R):
        return 2*R*np.sin(k)/k

    def _disc2(self, k, R):
        return 2*np.pi*R**2 * j1(k) / k

    def _disc3(self, k, R):
        return 4*np.pi*R**3 * (np.sin(k)/k - np.cos(k))/k**2

    def _psf_disc(self, k, params):
        return (1.0 + np.exp(-params[0]*params[1])) / (1.0 + np.exp(params[0]*(k - params[1])))

    def _setup_kvecs(self):
        kx = 2*np.pi*np.fft.fftfreq(self._shape_fft[2])[None,None,:]
        ky = 2*np.pi*np.fft.fftfreq(self._shape_fft[1])[None,:,None]
        kz = 2*np.pi*np.fft.fftfreq(self._shape_fft[0])[:,None,None]
        self._kvecs = np.array(np.broadcast_arrays(kz,ky,kx)).T
        self._klen = np.sqrt(kx**2 + ky**2 + kz**2)

    def _kparticle(self, pos, rad):
        kdotx = (pos * self._kvecs).sum(axis=-1)
        return self._disc3(self._klen*rad+1e-8, rad)*np.exp(-1.j*kdotx)

    def update_kspace_spheres(self, pos0, rad0, pos1, rad1):
        self.field_particles -= self._kparticle(pos0, rad0)
        self.field_particles += self._kparticle(pos1, rad1)

    def create_base_platonic_image(self):
        if self.index is None:
            raise AttributeError("Particle index has not been selected, call set_current_particle")

        self.field_particles = np.zeros(self._shape_fft, dtype='complex')

        for p0, r0 in zip(self._pos.reshape(-1,3), self._rad):
            pos = p0 - self._bounds[0]
            self.field_particles += self._kparticle(pos, r0)

    def create_bkg_field(self):
        self.field_bkg = self.poly.evaluate(self.state[self.b_bkg], self._slice)

    def create_final_image(self):
        particles = np.fft.ifftn(self.field_particles)
        kplatonic = np.fft.fftn(self.field_bkg * (1 - particles))
        kpsf = self._psf_disc(self._klen, self.state[self.b_psf])

        self.model_image = np.real(np.fft.ifftn(kplatonic * kpsf))
        return self.model_image

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
            pr = np.array(self.image.shape[::-1])
            center = (pr - pl)/2

        if (pl < 0).any() or (pr[::-1] > self.image.shape).any():
            return False

        # these variables map the buffer region back
        # into the large image in real space
        self._mask = ((pos > pl+rad[:,None]) & (pos < pr-rad[:,None])).all(axis=-1)
        self._center = center
        self._bounds = (pl, pr)
        self._slice = np.s_[pl[2]:pr[2], pl[1]:pr[1], pl[0]:pr[0]]
        self._shape = np.abs(pr - pl)[::-1]
        self._shape_fft = self._shape#self._shape + self.pad

        self._pos = pos[self._mask].flatten()
        self._rad = rad[self._mask]

        # these variables have to do with the comparison region
        # that is inside the buffer region
        inl = pl + self.pad/2
        inr = pr - self.pad/2
        self._cmp_bounds = (inl, inr)
        self._cmp2buffer = (np.s_[self.pad/2:-self.pad/2],)*3
        self._cmp_slice = np.s_[inl[2]:inr[2], inl[1]:inr[1], inl[0]:inr[0]]
        self._cmp_shape = np.abs(inr - inl)[::-1]
        self._cmp_mask = self.image[self._cmp_slice] > -10

        self._setup_kvecs()
        self.create_base_platonic_image()
        self.create_bkg_field()
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
            self.update_kspace_spheres(cpos0, rad0, cpos1, rad1)

        if bmask.any():
            self.create_bkg_field()

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

