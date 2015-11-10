"""
Plot the average positional / radius error vs fraction of self-diffusion time
"""
import pickle
import numpy as np
import scipy as sp
import scipy.ndimage as nd

import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import ImageGrid

from cbamf import const, runner, initializers
from cbamf.test import init
from cbamf.states import prepare_image
from cbamf.viz.util import COLORS
from cbamf.viz.plots import lbl

def set_image(state, cg, sigma):
    image = cg + np.random.randn(*cg.shape)*sigma
    image = np.pad(image, const.PAD, mode='constant', constant_values=const.PADVAL)
    state.set_image(image)
    state.sigma = sigma
    state.reset()

def perfect_platonic_global(N, R, scale=11, pos=None):
    if pos is None:
        # enforce the position to be on the pixel
        pos = np.array([int(N/2)*2]*3)/2.0 + 0.5

    if scale % 2 != 1:
        scale += 1

    pos = pos
    r = np.linspace(0, N, (N)*scale, endpoint=False)
    image = np.zeros((N*scale,)*3)
    x,y,z = np.meshgrid(*(r,)*3, indexing='ij')
    rvect = np.sqrt((x-pos[0])**2 + (y-pos[1])**2 + (z-pos[2])**2)
    image  = 1.0*(rvect <= R)

    vol_true = 4./3*np.pi*R**3
    vol_real = image.sum() / float(scale**3)
    print vol_true, vol_real, (vol_true - vol_real)
    return image

def perfect_platonic_per_pixel(N, R, scale=11, pos=None):
    """
    Create a perfect platonic sphere of a given radius R by supersampling by a
    factor scale on a grid of size N.  Scale must be odd.

    We are able to perfectly position these particles up to 1/scale. Therefore,
    let's only allow those types of shifts for now, but return the actual position
    used for the placement.
    """
    # enforce odd scale size
    if scale % 2 != 1:
        scale += 1

    if pos is None:
        # place the default position in the center of the grid
        pos = np.array([(N-1)/2.0]*3)

    # limit positions to those that are exact on the size 1./scale
    # positions have the form (d = divisions):
    #   p = N + m/d
    s = 1.0/scale

    i = pos.astype('int')
    p = i + s*((pos - i)/s).astype('int')
    pos = p + 1e-10 # unfortunately needed to break ties

    # make the output arrays
    image = np.zeros((N,)*3)
    x,y,z = np.meshgrid(*(xrange(N),)*3, indexing='ij')

    # for each real pixel in the image, integrate a bunch of superres pixels
    for x0,y0,z0 in zip(x.flatten(),y.flatten(),z.flatten()):

        # short-circuit things that are just too far away!
        ddd = np.sqrt((x0-pos[0])**2 + (y0-pos[1])**2 + (z0-pos[2])**2)
        if ddd > R + 2:
            image[x0,y0,z0] = 0.0
            continue

        # otherwise, build the local mesh and count the volume
        xp,yp,zp = np.meshgrid(
            *(np.linspace(i-0.5+s/2, i+0.5-s/2, scale, endpoint=True) for i in (x0,y0,z0)),
            indexing='ij'
        )
        ddd = np.sqrt((xp-pos[0])**2 + (yp-pos[1])**2 + (zp-pos[2])**2)
        vol = (1.0*(ddd < R) + 0.0*(ddd == R)).sum()
        image[x0,y0,z0] = vol / float(scale**3)

    #vol_true = 4./3*np.pi*R**3
    #vol_real = image.sum()
    #print vol_true, vol_real, (vol_true - vol_real)/vol_true

    return image, pos

def translate_fourier(image, dx):
    """ Translate an image in fourier-space with plane waves """
    N = image.shape[0]

    f = 2*np.pi*np.fft.fftfreq(N)
    kx,ky,kz = np.meshgrid(*(f,)*3, indexing='ij')
    kv = np.array([kx,ky,kz]).T

    q = np.fft.fftn(image)*np.exp(-1.j*(kv*dx).sum(axis=-1)).T
    return np.real(np.fft.ifftn(q))

def create_many_platonics(radius=5.0, scale=101, N=50):
    N = int(4*radius)

    platonics = []
    for i in xrange(N):
        goal = np.array([(N-1.0)/2]*3) + 2*np.random.rand(3)-1
        im, pos = perfect_platonic_per_pixel(N=N, R=radius, scale=scale, pos=goal)

        print i, goal, '=>', pos
        platonics.append((im, pos))
        pickle.dump(platonics, open('/tmp/platonics.pkl', 'w'))
    return platonics

def create_comparison_state(image, position, radius=5.0, snr=20):
    """
    Take a platonic image and position and create a state which we can
    use to sample the error for peri
    """
    # place that into a new image at the expected parameters
    s = init.create_single_particle_state(imsize=image.shape, sigma=1.0/snr,
            radius=radius, psfargs={'params': np.array([2.0, 1.0, 3.0]), 'error': 1e-6})
    s.obj.pos[0] = position
    s.set_image(s.psf.execute(image))
    return s

def crb(state):
    crb = []

    blocks = state.explode(state.block_all())
    for block in blocks:
        tc = np.sqrt(1.0/np.abs(state.fisher_information(blocks=[block])))
        crb.append(tc)

    return np.squeeze(np.array(crb))

def sample(state, im, noise, N=10, burn=10, sweeps=20):
    values, errors = [], []

    for i in xrange(N):
        print i, ' ',
        set_image(state, im, noise)
        h,l = runner.do_samples(state, sweeps, burn, quiet=True)

        h = np.array(h)
        values.append(h.mean(axis=0))
        errors.append(h.std(axis=0))

    print ''
    return np.array(values), np.array(errors)

def dorun(SNR=20, njitters=20, samples=10, noise_samples=10, sweeps=20, burn=10):
    """
    we want to display the errors introduced by pixelation so we plot:
        * CRB, sampled error vs exposure time

    a = dorun(ntimes=10, samples=5, noise_samples=5, sweeps=20, burn=8)
    """
    jitters = np.logspace(-6, np.log10(0.5), njitters)
    crbs, vals, errs, poss = [], [], [], []

    for i,t in enumerate(jitters):
        print '###### jitter', i, t

        for j in xrange(samples):
            print 'image', j, '|', 
            s,im,pos = zjitter(jitter=t)

            # typical image
            set_image(s, im, 1.0/SNR)
            crbs.append(crb(s))

            val, err = sample(s, im, 1.0/SNR, N=noise_samples, sweeps=sweeps, burn=burn)
            poss.append(pos)
            vals.append(val)
            errs.append(err)


    shape0 = (njitters, samples, -1)
    shape1 = (njitters, samples, noise_samples, -1)

    crbs = np.array(crbs).reshape(shape0)
    vals = np.array(vals).reshape(shape1)
    errs = np.array(errs).reshape(shape1)
    poss = np.array(poss).reshape(shape0)

    return  [crbs, vals, errs, poss, jitters]

def dist(a):
    return np.sqrt((a[...,:3]**2).sum(axis=-1)).mean(axis=-1)

def errs(val, pos):
    v,p = val, pos
    return np.sqrt(((v[...,:3] - p[:,:,None,:])**2).sum(axis=-1)).mean(axis=(1,2))

def doplot(prefix='/media/scratch/peri/does_matter/z-jitter', snrs=[20,50,200,500]):
    fig = pl.figure(figsize=(14,7))

    ax = fig.add_axes([0.43, 0.15, 0.52, 0.75])
    gs = ImageGrid(fig, rect=[0.05, 0.05, 0.25, 0.90], nrows_ncols=(2,1), axes_pad=0.25,
            cbar_location='right', cbar_mode='each', cbar_size='10%', cbar_pad=0.04)

    s,im,pos = zjitter(jitter=0.1, radius=5)
    nn = np.s_[:,:,im.shape[2]/2]

    figlbl, labels = ['A', 'B'], ['Reference', 'Difference']
    diff = (im - s.get_model_image()[s.inner])[nn]
    diffm = 0.1#np.abs(diff).max()
    im0 = gs[0].imshow(im[nn], vmin=0, vmax=1, cmap='bone_r')
    im1 = gs[1].imshow(diff, vmin=-diffm, vmax=diffm, cmap='RdBu')
    cb0 = pl.colorbar(im0, cax=gs[0].cax, ticks=[0,1])
    cb1 = pl.colorbar(im1, cax=gs[1].cax, ticks=[-diffm,diffm]) 
    cb0.ax.set_yticklabels(['0', '1'])
    cb1.ax.set_yticklabels(['-%0.1f' % diffm, '%0.1f' % diffm])

    for i in xrange(2):
        gs[i].set_xticks([])
        gs[i].set_yticks([])
        gs[i].set_ylabel(labels[i])
        #lbl(gs[i], figlbl[i])

    symbols = ['o', '^', 'D', '>']
    for i, snr in enumerate(snrs):
        c = COLORS[i]
        fn = prefix+'-snr-'+str(snr)+'.pkl'
        crb, val, err, pos, time = pickle.load(open(fn))

        if i == 0:
            label0 = r"$\rm{SNR} = %i$ CRB" % snr
            label1 = r"$\rm{SNR} = %i$ Error" % snr
        else:
            label0 = r"$%i$, CRB" % snr
            label1 = r"$%i$, Error" % snr

        ax.plot(time, dist(crb), '-', c=c, lw=3, label=label0)
        ax.plot(time, errs(val, pos), symbols[i], ls='--', lw=2, c=c, label=label1, ms=12)

    lbl(ax, 'D')
    ax.loglog()
    ax.set_ylim(1e-4, 1e0)
    ax.set_xlim(0, time[-1])
    ax.legend(loc='best', ncol=2, prop={'size': 18}, numpoints=1)
    ax.set_xlabel(r"$z$-scan NSR")
    ax.set_ylabel(r"Position CRB, Error")
    ax.grid(False, which='both', axis='both')
    ax.set_title(r"$z$-scan jitter")
