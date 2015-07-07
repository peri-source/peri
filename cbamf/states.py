import os
import numpy as np
from collections import OrderedDict
from .const import ZEROLOGPRIOR, PRIORCUT
from .util import Tile, amin, amax
from .priors import overlap

class State(object):
    def __init__(self, nparams, state=None, logpriors=None):
        self.nparams = nparams
        self.state = state if state is not None else np.zeros(self.nparams, dtype='double')
        self.stack = []
        self.logpriors = logpriors

    def _update_state(self, block, data):
        self.state[block] = data.astype(self.state.dtype)
        return True

    def push_update(self, block, data):
        curr = self.state[block].copy()
        self.stack.append((block, curr))
        self.update(block, data)

    def pop_update(self):
        block, data = self.stack.pop()
        self.update(block, data)

    def update(self, block, data):
        return self._update_state(block, data)

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

    def explode(self, block):
        inds = np.arange(block.shape[0])
        inds = inds[block]

        blocks = []
        for i in inds:
            tblock = self.block_none()
            tblock[i] = True
            blocks.append(tblock)
        return blocks

    def _grad_single_param(self, block, dl, action='ll'):
        # TODO -- add option for model_data grad / hess
        self.push_update(block, self.state[block]+dl)
        loglr = self.loglikelihood()
        self.pop_update()

        self.push_update(block, self.state[block]-dl)
        logll = self.loglikelihood()
        self.pop_update()

        return (loglr - logll) / (2*dl)

    def _hess_two_param(self, b0, b1, dl, action='ll'):
        self.push_update(b0, self.state[b0]+dl)
        self.push_update(b1, self.state[b1]+dl)
        logl_01 = self.loglikelihood()
        self.pop_update()
        self.pop_update()

        self.push_update(b0, self.state[b0]+dl)
        logl_0 = self.loglikelihood()
        self.pop_update()

        self.push_update(b1, self.state[b1]+dl)
        logl_1 = self.loglikelihood()
        self.pop_update()

        logl = self.loglikelihood()

        return (logl_01 - logl_0 - logl_1 + logl) / (dl**2)

    def get_model_data(self):
        pass

    def loglikelihood(self, state):
        loglike = self.dologlikelihood(state)
        if self.logpriors is not None:
            loglike += self.logpriors(state)
        return loglike

    def gradloglikelihood(self, dl=1e-3, blocks=None):
        if blocks is None:
            blocks = self.explode(self.block_all())
        grad = np.zeros(len(blocks))

        for i, b in enumerate(blocks):
            grad[i] = self._grad_single_param(b, dl)

        return grad

    def hessloglikelihood(self, dl=1e-3, jtj=False, blocks=None):
        if jtj:
            grad = self.gradloglikelihood(dl=dl, blocks=blocks)
            return grad.T[None,:] * grad[:,None]
        else:
            if blocks is None:
                blocks = self.explode(self.block_all())
            hess = np.zeros((len(blocks), len(blocks)))

            for i, bi in enumerate(blocks):
                for j, bj in enumerate(blocks[i+1:]):
                    J = j + i+1
                    thess = self._hess_two_param(bi, bj, dl)
                    hess[i,J] = thess
                    hess[J,i] = thess

            return hess

    def negloglikelihood(self, state):
        return -self.loglikelihood(state)

    def neggradloglikelihood(self, state):
        return -self.gradloglikelihood(state)


class LinearFit(State):
    def __init__(self, x, y, *args, **kwargs):
        super(LinearFit, self).__init__(*args, **kwargs)
        self.dx, self.dy = (np.array(i) for i in zip(*sorted(zip(x, y))))

    def plot(self, state):
        import pylab as pl
        pl.figure()
        pl.plot(self.dx, self.dy, 'o')
        pl.plot(self.dx, self.docalculate(state), '-')
        pl.show()

    def _calculate(self, state):
        return state.state[0]*self.dx + state.state[1]

    def dologlikelihood(self, state):
        return -((self._calculate(state) - self.dy)**2).sum()


class ConfocalImagePython(State):
    def __init__(self, image, obj, psf, ilm, zscale=1, offset=1,
            pad=24, sigma=0.1, doprior=True, constoff=False,
            varyn=False, allowdimers=False, nlogs=False, *args, **kwargs):
        self.pad = pad
        self.index = None
        self.sigma = sigma
        self.doprior = doprior
        self.constoff = constoff
        self.allowdimers = allowdimers
        self.nlogs = nlogs

        self.psf = psf
        self.ilm = ilm
        self.obj = obj
        self.zscale = zscale
        self.offset = offset
        self.N = self.obj.N

        self.param_dict = OrderedDict({
            'pos': 3*self.obj.N,
            'rad': self.obj.N,
            'typ': self.obj.N,
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
        self.b_typ = self.create_block('typ')
        self.b_psf = self.create_block('psf')
        self.b_ilm = self.create_block('ilm')
        self.b_off = self.create_block('off')
        self.b_zscale = self.create_block('zscale')

        self._build_state()
        self.set_image(image)

    def set_image(self, image):
        self.image = image.copy()
        self.image_mask = (image > -10).astype('float')
        self.image *= self.image_mask

        self._build_internal_variables()
        self._initialize()

    def get_model_image(self):
        return self.model_image * self.image_mask

    def get_true_image(self):
        return self.image * self.image_mask

    def _build_state(self):
        out = []
        for param in self.param_order:
            if param == 'pos':
                out.append(self.obj.get_params_pos())
            if param == 'rad':
                out.append(self.obj.get_params_rad())
            if param == 'typ':
                out.append(self.obj.get_params_typ())
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

        self._logprior = 0
        self._loglikelihood = 0
        self._loglikelihood_field = np.zeros(self.image.shape)

    def _initialize(self):
        if self.doprior:
            bounds = (np.array([0,0,0]), np.array(self.image.shape))
            self.nbl = overlap.HardSphereOverlapCell(self.obj.pos, self.obj.rad, self.obj.typ,
                    zscale=self.zscale, bounds=bounds, cutoff=2.2*self.obj.rad.max())
            self._logprior = self.nbl.logprior() + ZEROLOGPRIOR*(self.state[self.b_rad] < 0).any()

        self.psf.update(self.state[self.b_psf])
        self.obj.initialize(self.zscale)
        self.ilm.initialize()

        self._update_tile(*self._tile_global())

    def _tile_from_particle_change(self, p0, r0, t0, p1, r1, t1):
        psc = self.psf.get_support_size()
        rsc = self.obj.get_support_size()/2.0

        zsc = np.array([1.0/self.zscale, 1, 1])
        r0, r1 = zsc*r0, zsc*r1

        off0 = r0 + 2*psc + rsc
        off1 = r1 + 2*psc + rsc

        if t0[0] == 1 and t1[0] == 1:
            pl = np.floor(amin(p0-off0-1, p1-off1-1)).astype('int')
            pr = np.ceil (amax(p0+off0+1, p1+off1+1)).astype('int')
        if t0[0] != 1 and t1[0] == 1:
            pl = np.floor(p1-off1-1).astype('int')
            pr = np.ceil (p1+off1+1).astype('int')
        if t0[0] == 1 and t1[0] != 1:
            pl = np.floor(p0-off0-1).astype('int')
            pr = np.ceil (p0+off0+1).astype('int')
        if t0[0] != 1 and t1[0] != 1:
            pl = np.zeros(3)
            pr = np.array(self.image.shape)

        ipsc = np.ceil(psc)

        outer = Tile(pl, pr, 0, self.image.shape)
        inner = Tile(pl+ipsc, pr-ipsc, ipsc, np.array(self.image.shape)-ipsc)
        ioslice = tuple([np.s_[ipsc[i]:-ipsc[i]] for i in xrange(3)])
        return outer, inner, ioslice

    def _tile_global(self):
        outer = Tile(0, self.image.shape)
        inner = Tile(self.pad/2, np.array(self.image.shape)-self.pad/2)
        ioslice = (np.s_[self.pad/2:-self.pad/2],)*3
        return outer, inner, ioslice

    def _update_ll_field(self, data=None, slicer=np.s_[:]):
        if data is None:
            self._loglikelihood *= 0
            self._loglikelihood_field *= 0
            data = self.get_model_image()

        oldll = self._loglikelihood_field[slicer].sum()

        self._loglikelihood_field[slicer] = (
                -self.image_mask[slicer] * (data - self.image[slicer])**2 / (2*self.sigma**2)
                -np.log( np.sqrt(2*np.pi) * self.sigma )*self.nlogs
            )

        newll = self._loglikelihood_field[slicer].sum()
        self._loglikelihood += newll - oldll

    def _update_tile(self, otile, itile, ioslice):
        self._last_slices = (otile, itile, ioslice)

        self.obj.set_tile(otile)
        self.ilm.set_tile(otile)
        self.psf.set_tile(otile)

        islice = itile.slicer

        platonic = self.obj.get_field()
        if self.allowdimers:
            platonic = np.clip(platonic, 0, 1)

        if self.constoff:
            replacement = self.ilm.get_field() - self.offset*platonic
        else:
            replacement = self.ilm.get_field() * (1 - self.offset*platonic)
        replacement = self.psf.execute(replacement)

        self.model_image[islice] = replacement[ioslice]
        self._update_ll_field(replacement[ioslice], islice)

    def update(self, block, data):
        prev = self.state.copy()

        # TODO, instead, push the change in case we need to pop it
        self._update_state(block, data)

        pmask = block[self.b_pos].reshape(-1, 3).any(axis=-1)
        rmask = block[self.b_rad]
        tmask = block[self.b_typ]
        particles = np.arange(self.obj.N)[pmask | rmask | tmask]

        self._logprior = 0
        # if the particle was changed, update locally
        if len(particles) > 0:
            pos0 = prev[self.b_pos].copy().reshape(-1,3)[particles]
            rad0 = prev[self.b_rad].copy()[particles]
            typ0 = prev[self.b_typ].copy()[particles]

            pos = self.state[self.b_pos].copy().reshape(-1,3)[particles]
            rad = self.state[self.b_rad].copy()[particles]
            typ = self.state[self.b_typ].copy()[particles]

            # Do a bunch of checks to make sure that we can safetly modify
            # the image since that is costly and we would reject
            # this state eventually otherwise
            if (pos < 0).any() or (pos > np.array(self.image.shape)).any():
                self.state[block] = prev[block]
                return False

            tiles = self._tile_from_particle_change(pos0, rad0, typ0, pos, rad, typ)
            for tile in tiles[:2]:
                if (np.array(tile.shape) < 0).any():
                    self.state[block] = prev[block]
                    return False

            if self.doprior:
                self.nbl.update(particles, pos, rad, typ)
                self._logprior = self.nbl.logprior() + ZEROLOGPRIOR*(self.state[self.b_rad] < 0).any()

                if self._logprior < PRIORCUT:
                    self.nbl.update(particles, pos0, rad0, typ0)
                    self._logprior = self.nbl.logprior() + ZEROLOGPRIOR*(self.state[self.b_rad] < 0).any()

                    self.state[block] = prev[block]
                    return False

            if (typ0 == 0).all() and (typ == 0).all():
                return False

            # Finally, modify the image
            self.obj.update(particles, pos, rad, typ, self.zscale)
            self._update_tile(*tiles)
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

                if self.doprior:
                    bounds = (np.array([0,0,0]), np.array(self.image.shape))
                    tnbl = overlap.HardSphereOverlapCell(self.obj.pos, self.obj.rad, self.obj.typ,
                            zscale=self.zscale, bounds=bounds, cutoff=2.2*self.obj.rad.max())

                    if tnbl.logprior() < PRIORCUT:
                        self.state[block] = prev[block]
                        return False

                    self.nbl = tnbl
                    self._logprior = self.nbl.logprior() + ZEROLOGPRIOR*(self.state[self.b_rad] < 0).any()

                self.obj.initialize(self.zscale)
                self._update_tile(*self._tile_global())

            if docalc:
                self._update_tile(*self._tile_global())

        return True

    def isactive(self, particle):
        return self.state[self.block_particle_typ(particle)] == 1

    def blocks_particle(self, index):
        p_ind, r_ind = 3*index, 3*self.obj.N + index

        blocks = []
        for t in xrange(p_ind, p_ind+3):
            blocks.append(self.block_range(t, t+1))
        blocks.append(self.block_range(r_ind, r_ind+1))
        return blocks

    def block_particle_pos(self, index):
        a = self.block_none()
        a[3*index:3*index+3] = True
        return a

    def block_particle_rad(self, index):
        a = self.block_none()
        a[3*self.obj.N + index] = True
        return a

    def block_particle_typ(self, index):
        a = self.block_none()
        a[4*self.obj.N + index] = True
        return a

    def create_block(self, typ='all'):
        return self.block_range(*self._block_offset_end(typ))

    def _block_offset_end(self, typ='pos'):
        index = self.param_order.index(typ)
        off = sum(self.param_lengths[:index])
        end = off + self.param_lengths[index]
        return off, end

    def loglikelihood(self):
        return self._logprior + self._loglikelihood
