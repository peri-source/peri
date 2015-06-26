import os
import numpy as np
from collections import OrderedDict
from .util import Tile
from .priors import overlap

class State(object):
    def __init__(self, nparams, state=None):
        self.nparams = nparams
        self.state = state if state is not None else np.zeros(self.nparams, dtype='double')
        self.stack = []

    def update(self, block, data):
        return self._update_state(block, data)

    def _update_state(self, block, data):
        self.state[block] = data.astype(self.state.dtype)

    def _push_update(self, block, data):
        curr = self.state[block].copy()
        self.stack.append((block, curr))
        self.update(block, data)

    def _pop_update(self):
        block, data = self.stack.pop()
        self.update(block, data)

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

class ConfocalImagePython(State):
    def __init__(self, image, obj, psf, ilm, zscale=1, offset=1,
            pad=16, sigma=0.1, doprior=True, *args, **kwargs):
        self.pad = pad
        self.index = None
        self.sigma = sigma
        self.doprior = doprior

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

        self._build_state()
        self.set_image(image)

    def set_image(self, image):
        self.image = image
        self.image_mask = (image > -10).astype('float')
        self.image *= self.image_mask

        self._build_internal_variables()
        self._initialize()

    def get_model_image(self):
        return self.model_image * self.image_mask

    def _build_state(self):
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

        self.state = np.hstack(out).astype('float')

    def _build_internal_variables(self):
        self.model_image = np.zeros_like(self.image)
        self._loglikelihood_field = -self.image_mask*self.image**2 / (2*self.sigma**2)
        self._loglikelihood = self._loglikelihood_field.sum()
        self._logprior = 0

    def _initialize(self):
        self.psf.update(self.state[self.b_psf])
        self.obj.initialize(self.zscale)
        self.ilm.initialize()

        bounds = (np.array([0,0,0]), np.array(self.image.shape))
        self.nbl = overlap.HardSphereOverlapCell(self.obj.pos, self.obj.rad,
                zscale=self.zscale, bounds=bounds, cutoff=2.2*self.obj.rad.max())

        if self.doprior:
            self._logprior = self.nbl.logprior() + -1e100*(self.state[self.b_rad] < 0).any()
        self._update_tile(*self._tile_global())

    def _tile_from_particle_change(self, p0, r0, p1, r1):
        psc = self.psf.get_support_size()/2

        zsc = np.array([1.0/self.zscale, 1, 1])
        r0, r1 = zsc*r0, zsc*r1
        pl = np.round(np.vstack(
                [p0-r0-3-self.pad/2-psc, p1-r1-3-self.pad/2-psc]
            ).min(axis=0)).astype('int')
        pr = np.round(np.vstack(
                [p0+r0+3+self.pad/2+psc+1, p1+r1+3+self.pad/2+psc+1]
            ).max(axis=0)).astype('int')

        outer = Tile(pl, pr, 0, self.image.shape)
        inner = Tile(pl+self.pad/2, pr-self.pad/2, self.pad/2, np.array(self.image.shape)-self.pad/2)
        ioslice = (np.s_[self.pad/2:-self.pad/2],)*3
        return outer, inner, ioslice

    def _tile_global(self):
        outer = Tile(0, self.image.shape)
        inner = Tile(self.pad/2, np.array(self.image.shape)-self.pad/2)
        ioslice = (np.s_[self.pad/2:-self.pad/2],)*3
        return outer, inner, ioslice

    def _update_tile(self, otile, itile, ioslice):
        self._last_slices = (otile, itile, ioslice)

        self.obj.set_tile(otile)
        self.ilm.set_tile(otile)
        self.psf.set_tile(otile)

        islice = itile.slicer
        oldll = self._loglikelihood_field[islice].sum()

        replacement = self.ilm.get_field() * (1 - self.offset*self.obj.get_field())
        replacement = self.psf.execute(replacement)

        self.model_image[islice] = replacement[ioslice]
        self._loglikelihood_field[islice] = -self.image_mask[islice]*(replacement[ioslice] - self.image[islice])**2 / (2*self.sigma**2)

        newll = self._loglikelihood_field[islice].sum()
        self._loglikelihood += newll - oldll

    def update(self, block, data):
        prev = self.state.copy()

        # TODO, instead, push the change in case we need to pop it
        self._update_state(block, data)

        pmask = block[self.b_pos].reshape(-1, 3)
        rmask = block[self.b_rad]
        particles = np.arange(self.obj.N)[pmask.any(axis=-1) | rmask]

        # if the particle was changed, update locally
        if len(particles) > 0:
            pos0 = prev[self.b_pos].copy().reshape(-1,3)[particles]
            rad0 = prev[self.b_rad].copy()[particles]

            pos = self.state[self.b_pos].copy().reshape(-1,3)[particles]
            rad = self.state[self.b_rad].copy()[particles]

            # TODO - check why we need to have obj.update here?? should
            # only be necessary before _update_tile
            self.obj.update(particles, pos, rad, self.zscale)
            self.nbl.update(particles, pos, rad)

            if self.doprior:
                self._logprior = self.nbl.logprior() + -1e100*(self.state[self.b_rad] < 0).any()

            # check all the priors before actually going for an update
            # if it is too small, don't both and return False
            # This needs to be more general with pop and push
            if self._logprior < -1e90:
                self.obj.update(particles, pos0, rad0, self.zscale)
                self.nbl.update(particles, pos0, rad0)
                self.state[block] = prev[block]
                return False

            self._update_tile(*self._tile_from_particle_change(pos0, rad0, pos, rad))
        else:
            docalc = False

            # if the psf was changed, update globally
            if block[self.b_psf].any():
                self.psf.update(self.state[self.b_psf])
                docalc = True

            # update the background if it has been changed
            if block[self.b_ilm].any():
                self.ilm.update(self.state[self.b_ilm])
                docalc = True

            # we actually don't have to do anything if the offset is changed
            if block[self.b_off].any():
                self.offset = self.state[self.b_off]
                docalc = True

            if block[self.b_zscale].any():
                self.zscale = self.state[self.b_zscale][0]
                self._initialize()

            if docalc:
                self._update_tile(*self._tile_global())

    def blocks_particle(self, index):
        p_ind, r_ind = 3*index, 3*self.obj.N + index

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
        self._push_update(block, self.state[block]+dl)
        loglr = self.loglikelihood()
        self._pop_update()

        self._push_update(block, self.state[block]-dl)
        logll = self.loglikelihood()
        self._pop_update()

        return (loglr - logll) / (2*dl)

    def _hess_two_param(self, b0, b1, dl):
        self._push_update(b0, self.state[b0]+dl)
        self._push_update(b1, self.state[b1]+dl)
        logl_01 = self.loglikelihood()
        self._pop_update()
        self._pop_update()

        self._push_update(b0, self.state[b0]+dl)
        logl_0 = self.loglikelihood()
        self._pop_update()

        self._push_update(b1, self.state[b1]+dl)
        logl_1 = self.loglikelihood()
        self._pop_update()

        logl = self.loglikelihood()

        return (logl_01 - logl_0 - logl_1 + logl) / (dl**2)

    def gradloglikelihood(self, dl=1e-3):
        blocks = self.explode(self.block_all())
        grad = np.zeros(self.nparams)

        for i in xrange(self.nparams):
            grad[i] = self._grad_single_param(blocks[i], dl)

        return grad

    def hessloglikelihood(self, dl=1e-3, jtj=False):
        if jtj:
            grad = self.gradloglikelihood(dl=dl)
            return grad.T[None,:] * grad[:,None]
        else:
            blocks = self.explode(self.block_all())
            hess = np.zeros((self.nparams, self.nparams))

            for i in xrange(self.nparams):
                for j in xrange(i, self.nparams):
                    thess = self._hess_two_param(blocks[i], blocks[j], dl)
                    hess[i,j] = thess
                    hess[j,i] = thess

            return hess

    def loglikelihood(self):
        return self._logprior + self._loglikelihood
