import os
import numpy as np
from collections import OrderedDict
from .util import Tile

def loadtest():
    import pickle
    itrue, xstart, rstart, pstart, ipure = pickle.load(open("/media/scratch/bamf/bamf_ic_16.pkl", 'r'))
    ilm = np.zeros((3,3,3))
    ilm[0,0,0] = 1
    state = np.hstack([xstart.flatten(), rstart, pstart, ilm.ravel(), np.ones(1), np.zeros(1)])
    return ConfocalImagePython(len(rstart), itrue, pad=32, order=(3,3,3), state=state, fftw_planning_level=FFTW_PLAN_SLOW)

class State(object):
    def __init__(self, nparams, state=None):
        self.nparams = nparams
        self.state = state if state is not None else np.zeros(self.nparams, dtype='double')
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

class ConfocalImagePython(State):
    def __init__(self, image, obj, psf, ilm, zscale=1, offset=0,
            pad=16, sigma=0.1, *args, **kwargs):
        self.image = image
        self.pad = pad
        self.index = None
        self.sigma = sigma

        self.psf = psf
        self.ilm = ilm  
        self.obj = obj 
        self.zscale = zscale
        self.offset = offset
        self.N = self.obj.N

        self.param_dict = OrderedDict({
            'pos': 3*self.obj.N,
            'rad': self.obj.N,
            'typ': 0,
            'psf': len(self.psf.get_params()),
            'ilm': len(self.ilm.get_params()),
            'off': 1,
            'zscale': 1
        })

        self.param_order = ['pos', 'rad', 'typ', 'psf', 'ilm', 'off', 'zscale']
        self.param_lengths = [self.param_dict[k] for k in self.param_order]

        total_params = sum(self.param_lengths)
        super(ConfocalImagePython, self).__init__(nparams=total_params, *args, **kwargs)

        self.b_pos = self.create_block('pos')
        self.b_rad = self.create_block('rad')
        self.b_psf = self.create_block('psf')
        self.b_ilm = self.create_block('ilm')
        self.b_off = self.create_block('off')
        self.b_zscale = self.create_block('zscale')

        self.build_state()

    def build_state(self):
        out = []
        for param in self.param_order:
            if param == 'pos':
                out.append(self.obj.get_params_pos())
            if param == 'rad':
                out.append(self.obj.get_params_rad())
            if param == 'psf':
                out.append(self.psf.get_params())
            if param == 'ilm':
                out.append(self.ilm.get_params())
            if param == 'off':
                out.append(self.offset)
            if param == 'zscale':
                out.append(self.zscale)

        self.state = np.hstack(out)

        self.initialize()
        self.update_psf()

    def initialize(self):
        self.obj.initialize(self.zscale)
        self.ilm.initialize()

    def update_ilm(self):
        self.ilm.update(self.state[self.b_ilm])
    
    def update_psf(self):
        self.psf.update(self.state[self.b_psf])

    def create_final_image(self):
        illumination = self.ilm.get_field() * (1 - self.obj.get_field())
        self.model_image = self.psf.execute(illumination) + self.offset
        return self.model_image

    def create_differences(self):
        sl = self.tile_cmp.slicer
        return (
            self.model_image[self._cmp_region][self._cmp_mask]-
            self.image[sl][self._cmp_mask]
        )

    def set_current_particle(self, index=None, sub_image_size=None):
        """
        We must set up the following structure:
        +-----------------------------+
        |        Buffer Region        |
        |(ilm field + other particles)|
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
        self.tile = Tile(pl, pr)

        # these variables have to do with the comparison region
        # that is inside the buffer region
        self.tile_cmp = Tile(pl + self.pad/2, pr - self.pad/2)
        self._cmp_region = (np.s_[self.pad/2:-self.pad/2],)*3
        self._cmp_mask = self.image[self.tile_cmp.slicer] > -10

        self.obj.set_tile(self.tile)
        self.ilm.set_tile(self.tile)
        self.psf.set_tile(self.tile)
        return True

    def update(self, block, data):
        self._update_state(block, data)

        pmask = block[self.b_pos].reshape(-1, 3)
        rmask = block[self.b_rad]
        particles = np.arange(self.obj.N)[pmask.any(axis=-1) | rmask]

        pos = self.state[self.b_pos].copy().reshape(-1,3)[particles]
        rad = self.state[self.b_rad].copy()[particles]

        if len(pos) > 0 and len(rad) > 0:
            self.obj.update(particles, pos, rad, self.zscale)

        # if the psf was changed, update
        if block[self.b_psf].any():
            self.update_psf()

        # update the background if it has been changed
        if block[self.b_ilm].any():
            self.update_ilm()

        # we actually don't have to do anything if the offset is changed
        if block[self.b_off].any():
            self.offset = self.state[self.b_off]

        if block[self.b_zscale].any():
            self.zscale = self.state[self.b_zscale]
            self.initialize()

    def blocks_particle(self):
        if self.index is None:
            raise AttributeError("No particle selected, run set_current_particle")

        p_ind, r_ind = 3*self.index, 3*self.obj.N + self.index

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
                for i in xrange(self.obj.N):
                    self.set_current_particle(i)
                    blocks = self.blocks_particle()[:-1]

                    for block in blocks:
                        grad.append(self._grad_single_param(block, dl))

            if pg == 'rad':
                for i in xrange(self.obj.N):
                    self.set_current_particle(i)
                    block = self.blocks_particle()[-1]
                    grad.append(self._grad_single_param(block, dl))

            if pg == 'psf' or pg == 'ilm' or pg == 'off' or pg == 'zscale':
                self.set_current_particle()
                blocks = self.explode(self.create_block(pg))

                for block in blocks:
                    grad.append(self._grad_single_param(block, dl))

        return np.array(grad)

    def loglikelihood(self):
        self.create_final_image()
        return -(self.create_differences()**2).sum() / self.sigma**2
