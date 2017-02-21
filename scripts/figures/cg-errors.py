import copy
import pickle
import numpy as np
from trackpy import locate
from collections import OrderedDict

from peri import util, runner, states, initializers
from peri.test import nbody
from peri.comp import ilms, objs, psfs, exactpsf
from peri.test import analyze

FIXEDSS = [31,17,29]

def totiff(state):
    q = initializers.normalize(state.model_image)
    return (q*255).astype('uint8')

def trackpy(state, magic=165):
    image = totiff(state)
    diameter = int(2*state.state[state.b_rad].mean())
    diameter -= 1 - diameter % 2
    minmass = 100. if magic is None else magic*(diameter/2)**3
    out = locate(image, diameter=diameter, invert=True, minmass=minmass)
    return np.vstack([out.z, out.y, out.x]).T

def create_image(N=128, size=64, radius=6.0, pad=16, identity=False, polydispersity=0.0):
    blank = np.zeros((size,)*3)

    pos, rad, tile = nbody.initialize_particles(
        N, radius=radius, tile=util.Tile(blank.shape), polydispersity=polydispersity
    )
    sim = nbody.BrownianHardSphereSimulation(pos, rad, tile)
    sim.relax(2000)
    sim.step(5000)
    sim.relax(2000)

    slab_zpos = -radius
    if not identity:
        s = runner.create_state(
            blank, pos, rad, slab=slab_zpos, sigma=0.00001,
            stateargs={'pad': pad, 'offset': 0.18},
            psftype='cheb-linescan-fixedss', psfargs={
                'zslab': 10., 'cheb_degree': 6, 'cheb_evals': 8,
                'support_size': FIXEDSS,
            },
            ilmtype='barnesleg2p1dx', ilmargs={'order': (1,1,3), 'npts': (30,10,5)}
        )
        s.ilm.randomize_parameters(ptp=0.4, vmax=1.0, fourier=False)
    else:
        s = runner.create_state(
            blank, pos, rad, slab=slab_zpos, sigma=0.00001,
            stateargs={'pad': pad, 'offset': 0.18},
            psftype='identity', ilmtype='leg2p1d',
        )

    s.reset()
    s.model_to_true_image()
    return s

def table():
    s = create_image(identity=True)

    lpoly = [0.0, 0.0, 0.01, 0.05, 0.10]
    dnames = ['0.0', '0.0', '0.01', '0.05', '0.10']

    lilms = [
        ilms.LegendrePoly2P1D(shape=s.ilm.shape, order=(1,1,1)),
        ilms.LegendrePoly2P1D(shape=s.ilm.shape, order=(3,3,3)),
        ilms.BarnesStreakLegPoly2P1DX3(shape=s.ilm.shape, order=(1,1,1), npts=(10,5)),
        ilms.BarnesStreakLegPoly2P1DX3(shape=s.ilm.shape, order=(1,1,2), npts=(30,10)),
        ilms.BarnesStreakLegPoly2P1DX3(shape=s.ilm.shape, order=s.ilm.order,npts=(30,10,5)),
    ]
    lnames = [
        r'Legendre 2+1D (0,0,0)',
        r'Legendre 2+1D (2,2,2)',
        r'Barnes (10, 5), $N_z=1$',
        r'Barnes (30, 10), $N_z=2$',
        r'Barnes (30, 10, 5), $N_z=3$',
    ]

    lpsfs = [
        psfs.IdentityPSF(shape=s.psf.shape, params=np.array([0.0])),
        psfs.AnisotropicGaussian(shape=s.psf.shape, params=(2.0, 1.0, 3.0)),
        psfs.Gaussian4DLegPoly(shape=s.psf.shape, order=(3,3,3)),
        exactpsf.FixedSSChebLinePSF(
            shape=s.psf.shape, zrange=(0, s.psf.shape[0]), cheb_degree=3, cheb_evals=6,
            support_size=FIXEDSS, zslab=10., cutoffval= 1./255,
            measurement_iterations=3,
        ),
        exactpsf.FixedSSChebLinePSF(
            shape=s.psf.shape, zrange=(0, s.psf.shape[0]), cheb_degree=6, cheb_evals=8,
            support_size=FIXEDSS, zslab=10., cutoffval= 1./255,
            measurement_iterations=3,
        ),
    ]
    pnames = [
        r'Identity',
        r'Gaussian$(x,y)$',
        r'Gaussian$(x,y,z,z^{\prime})$',
        r'Cheby linescan (3,6)',
        r'Cheby linescan (6,8)',
    ]

    results = OrderedDict()

    for i in xrange(len(lpoly)):
        print dnames[i], lnames[i], pnames[i]
        poly = lpoly[i]
        ilm = lilms[i]
        psf = lpsfs[i]

        s = create_image(polydispersity=poly)
        s.set_ilm(ilm)
        s.set_psf(psf)
        if isinstance(s.ilm, ilms.BarnesStreakLegPoly2P1DX3):
            s.ilm.randomize_parameters(ptp=0.4, vmax=1.0, fourier=False)
        s.reset()
        s.model_to_true_image()

        pos1 = trackpy(s)
        pos0 = s.obj.pos.copy()

        s.obj.set_pos_rad(pos1, s.obj.rad.mean()*np.ones(pos1.shape[0]))
        s.reset()

        slicer = s.get_difference_image().shape[0]/2
        results[i] = (
            dnames[i], lnames[i], pnames[i],
            s.get_difference_image()[slicer].copy(),
            pos0, pos1,
        )

    return results

def scores(results):
    tmp = copy.copy(results)

    scores = []
    for k,v in tmp.iteritems():
        ind = analyze.nearest(v[5], v[4])
        scores.append((
            v[0], v[1], v[2], v[3],
            np.median(np.sqrt(((v[4][ind] - v[5])**2).sum(axis=-1))),
            float(len(v[5])) / len(v[4])
        ))
    return scores

def numform(x):
    p = int(np.floor(np.log10(x)))
    n = x / 10**p
    return "{:1.2f} ({:d})".format(n, p)

def numform2(x):
    return "{:0.5f}".format(x)

def print_table(tables):
    outstr = (
        '\\begin{center}\n'
        '\\begin{table}\n'
        '\\begin{tabular}{| l | l | l | c | c |}\n'
        '\\hline\n'
        'Polydispersity &\n'
        'Illumination field &\n'
        'Point spread function &\n'
        'Position error $\\langle | \\vec{r}_{\\rm{fit}} - \\vec{r}_{\\rm{true}} | \\rangle$ &\n'
        '\% Identified \\\\ \\hline \\hline\n'
    )

    ss = max([len(i) for row in tables for i in row[:3]]) + 3

    for row in tables:
        v = [numform2(i) for i in row[-2:]]
        outstr += "{:<{}s} & {:<{}s} & {:<{}s} & {:s} & {:s} \\\\ \\hline\n".format(
            row[0], ss, row[1], ss, row[2], ss, *v)

    outstr += (
        '\\end{tabular}\n'
        '\\caption{\\textbf{Crocker-Grier featuring errors.}}\n'
        '\\label{table:cg-complexity}\n'
        '\\end{table}\n'
        '\\end{center}\n'
    )
    return outstr
