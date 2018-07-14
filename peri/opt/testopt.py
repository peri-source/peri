# ~~~ testing stuff; delete me ~~~
# from peri.opt import opttest
import sys

from peri import util, states, models
from peri.comp import psfs, objs, ilms, GlobalScalar, ComponentCollection
# import peri.opt.optimize as opt
from peri.test import nbody

sys.path.append('./')
import opttest
from optengine import *
import optimize as opt


# do_levmarq_n_directions : ??


# ~ Things that might be added ~~
def limit_particles(st, inds):
    params = st.param_particle(inds)
    left_lim = st.oshape.translate(-st.pad).l.tolist()
    right_lim = st.oshape.translate(-st.pad).r
    vals = {'z':0, 'y':1, 'x':2}
    param_ranges = []
    for p in params:
        k = p.split('-')[-1]
        if k in vals.keys():
            i = vals[k]
            param_ranges.append([left_lim[i], right_lim[i]])
        else:
            param_ranges.append([-np.inf, np.inf])
    return np.array(param_ranges)


# TODO make versions of the convenience functions:
# Needs rts tags,
# do_levmarq:

def create_state_optimizer(st, param_names, **kwargs):
    optobj = OptImageState(st, param_names)
    lm = LMOptimizer(o, **kwargs)
    return lm


def optimize_parameters(st, param_names, **kwargs):
    lm = create_state_optimizer(st, param_names, **kwargs)
    l.optimize()


# do_levmarq_particles :
def optimize_particles(st, particle_indices, include_rad=True, **kwargs):
    param_names = (st.param_particle(particle_indices) if include_rad
                   else st.param_particle_pos(inds))
    optimize_parameters(st, param_names)



# do_levmarq_all_particle_groups :
def particles_generator(st, groups, include_rad=True, rts=True, dl=3e-6,
                        **kwargs):
    """does stuff"""
    for g in groups:
        param_names = (st.param_particle(g) if include_rad else
                       st.param_particle_pos(g))
        optobj = OptImageState(st, param_names, dl=dl, rts=rts)
        yield LMOptimizer(optobj, **kwargs)


# Write these with burn() in mind.
# burn needs a GroupedOptimizer, comprised of two other GroupedOptimizer's,
# one of which is a optimizer with globals, one of which is another grouped
# optimizer of particle groups
def create_particle_groups_optimizer(st, groups, **kwargs)
    optimizer_generator = particles_generator(st, groups, **kwargs)
    # param_ranges=limit_particles(st, g) for particles generator?
    return GroupedOptimizer(optimizer_generator)


def optimize_particle_groups(st, groups, **kwargs):
    """Levenberg-Marquardt optimization for groups of particles.

    Parameters
    ---------
    st : `peri.states.ImageState`
    groups : list
        A list of groups of particles. Each

    Other Parameters
    ----------------
    """
    grouped_optimizer = create_particle_groups_optimizer(st, groups, **kwargs)
    grouped_optimizer.optimizer()


# burn :
def burn_generator(st, **kwargs):
    for i in range(n_iter):
        # 1. Optimizer globals
        global_names = opt.name_globals(st)  # FIXME opt
        yield create_state_optimizer(st, global_names, **kwargs)
        # 2. Particles FIXME opt
        doshift = (i % 2) != 0
        particle_groups = opt.separate_particles_into_groups(
            st, doshift=doshift) # **kwargs? FIXME
        yield create_particle_groups_optimizer(st, particle_groups, **kwargs)

# -- still need a burn function
# finish :
# fit_comp :


# Test an OptObj:
def test_object():
    f = opttest.increase_model_dimension(opttest.rosenbrock)
    o = OptFunction(f, f(np.array([1.0, 1.0])), np.zeros(2), dl=1e-7)
    l = LMOptimizer(o, maxiter=1)
    l.optimize()
    return f, o, l


# Test an ImageState:
def test_simple_state():
    im = util.RawImage('../../scripts/bkg_test.tif')
    bkg_const = ilms.LegendrePoly3D(order=(1,1,1))
    st = states.ImageState(im, [bkg_const], mdl=models.SmoothFieldModel())
    o = OptImageState(st, st.params)
    l = LMOptimizer(o, maxiter=3)
    l.optimize()
    return st, o, l


def make_complex_state():
    im = util.NullImage(shape=(32,)*3)
    pos, rad, tile = nbody.create_configuration(3, im.tile)
    P = ComponentCollection([
        objs.PlatonicSpheresCollection(pos, rad),
        objs.Slab(2)
    ], category='obj')

    H = psfs.AnisotropicGaussian()
    I = ilms.BarnesStreakLegPoly2P1D(
        npts=(25, 13, 3), zorder=2, local_updates=False)
    B = ilms.LegendrePoly2P1D(order=(3, 1, 1), category='bkg', constval=0.01)
    C = GlobalScalar('offset', 0.0)
    I.randomize_parameters()

    st = states.ImageState(
        im, [B, I, H, P, C], pad=16, model_as_data=True, sigma=1e-4)
    return st


def test_complex_state():
    st = make_complex_state()

    true_error = st.error
    true_values = np.copy(st.get_values(st.params))
    # randomize:
    st.update(st.params, true_values + np.random.randn(true_values.size)*1e-1)
    initial_error = st.error
    o = OptImageState(st, st.params)
    # l = LMOptimizer(o, maxiter=3, damp=1e3)
    l = LMOptimizer(o, maxiter=1, damp=1e3)
    l.optimize()
    print("Initial error:\t{}\nCurrent error:\t{}\nTrue error:\t{}".format(
        initial_error, st.error, true_error))
    return st, o, l


def test_grouped_om():
    st = make_complex_state()
    go = GroupedOptimizer(particles_generator(st, region_size=2))
    go.optimize()
    return st


if __name__ == '__main__1':
    # also works for image state, w/o decimation...
    log.set_level('debug')
    # f, o, l = test_object()
    # st, o, l = test_simple_state()
    # st, o, l = test_complex_state()
    st = test_grouped_om()

    # so: Make it work for decimation, make opt procedures.
