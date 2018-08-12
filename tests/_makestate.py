import numpy as np; np.random.seed(0)

from peri import states, util
from peri.comp import objs, ilms, psfs, comp


def make_state(**kwargs):
    statemaker = MakeState(**kwargs)
    return statemaker.make_state()


class MakeState(object):
    def __init__(self, imshape=(50, 128, 128), n_particles=800, radius=5.,
                 particle_polydispersity=0.1):
        self.imshape = np.array(imshape)
        self.n_particles = n_particles
        self.radius = radius
        self.particle_polydispersity = particle_polydispersity

    def make_state(self):
        obj = self._make_object()
        ilm = self._make_ilm()
        bkg = self._make_bkg()
        psf = self._make_psf()
        c = comp.GlobalScalar(value=1.0, name='offset')

        im = util.NullImage(shape=self.imshape)
        state = states.ImageState(im, [obj, ilm, bkg, psf, c])
        return state

    def _make_object(self):
        positions = np.random.rand(self.n_particles, self.imshape.size)
        positions *= self.imshape.reshape(1, -1)
        radii = ((np.random.randn(self.n_particles) *
                  self.particle_polydispersity + 1 ) * self.radius)
        obj = objs.PlatonicSpheresCollection(positions, radii)
        return obj

    def _make_ilm(self):
        ilm = ilms.LegendrePoly3D(order=(1, 1, 1))
        return ilm

    def _make_bkg(self):
        bkg = ilms.LegendrePoly3D(order=(1, 1, 1), category='bkg')
        return bkg

    def _make_psf(self):
        psf = psfs.AnisotropicGaussian()
        return psf

