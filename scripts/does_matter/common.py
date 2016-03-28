import pickle
import string
import numpy as np

import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import ImageGrid

from cbamf import const, runner
from cbamf.test import init
from cbamf.viz.base import COLORS
from cbamf.viz.plots import lbl

def pad(image, size, val=0):
    return np.pad(image, size, mode='constant', constant_values=val)

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

def set_image(state, cg, sigma):
    image = cg + np.random.randn(*cg.shape)*sigma
    image = np.pad(image, state.pad, mode='constant', constant_values=const.PADVAL)
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

def perfect_platonic_per_pixel(N, R, scale=11, pos=None, zscale=1.0, returnpix=None):
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
    f = zscale**2

    i = pos.astype('int')
    p = i + s*((pos - i)/s).astype('int')
    pos = p + 1e-10 # unfortunately needed to break ties

    # make the output arrays
    image = np.zeros((N,)*3)
    x,y,z = np.meshgrid(*(xrange(N),)*3, indexing='ij')

    # for each real pixel in the image, integrate a bunch of superres pixels
    for x0,y0,z0 in zip(x.flatten(),y.flatten(),z.flatten()):

        # short-circuit things that are just too far away!
        ddd = np.sqrt(f*(x0-pos[0])**2 + (y0-pos[1])**2 + (z0-pos[2])**2)
        if ddd > R + 4:
            image[x0,y0,z0] = 0.0
            continue

        # otherwise, build the local mesh and count the volume
        xp,yp,zp = np.meshgrid(
            *(np.linspace(i-0.5+s/2, i+0.5-s/2, scale, endpoint=True) for i in (x0,y0,z0)),
            indexing='ij'
        )
        ddd = np.sqrt(f*(xp-pos[0])**2 + (yp-pos[1])**2 + (zp-pos[2])**2)

        if returnpix is not None and returnpix == [x0,y0,z0]:
            outpix = 1.0 * (ddd < R)

        vol = (1.0*(ddd < R) + 0.0*(ddd == R)).sum()
        image[x0,y0,z0] = vol / float(scale**3)

    #vol_true = 4./3*np.pi*R**3
    #vol_real = image.sum()
    #print vol_true, vol_real, (vol_true - vol_real)/vol_true

    if returnpix:
        return image, pos, outpix
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
    size = int(4*radius)

    platonics = []
    for i in xrange(N):
        goal = np.array([(size-1.0)/2]*3) + 2*np.random.rand(3)-1
        im, pos = perfect_platonic_per_pixel(N=size, R=radius, scale=scale, pos=goal)

        print i, goal, '=>', pos
        platonics.append((im, pos))
        pickle.dump(platonics, open('/tmp/platonics.pkl', 'w'))
    return platonics

def dist(a):
    return np.sqrt((a[...,:3]**2).sum(axis=-1)).mean(axis=-1)

def errs(val, pos):
    v,p = val, pos
    return np.sqrt(((v[...,:3] - p[:,:,None,:])**2).sum(axis=-1)).mean(axis=(1,2))

def snr_labels(i):
    pass

figlbl = [i.upper() for i in string.ascii_lowercase]

def doplot(image0, image1, xs, crbs, errors, labels, diff_image_scale=0.1,
        dolabels=True, multiple_crbs=True, xlim=None, ylim=None, highlight=None,
        detailed_labels=False, xlabel="", title=""):
    """
    Standardizing the plot format of the does_matter section.  See any of the
    accompaning files to see how to use this generalized plot.

    image0 : ground true
    image1 : difference image
    xs : list of x values for the plots
    crbs : list of lines of values of the crbs
    errors : list of lines of errors
    labels : legend labels for each curve
    """
    if len(crbs) != len(errors) or len(crbs) != len(labels):
        raise IndexError, "lengths are not consistent"

    fig = pl.figure(figsize=(14,7))

    ax = fig.add_axes([0.43, 0.15, 0.52, 0.75])
    gs = ImageGrid(fig, rect=[0.05, 0.05, 0.25, 0.90], nrows_ncols=(2,1), axes_pad=0.25,
            cbar_location='right', cbar_mode='each', cbar_size='10%', cbar_pad=0.04)

    diffm = diff_image_scale*np.ceil(np.abs(image1).max()/diff_image_scale)

    im0 = gs[0].imshow(image0, vmin=0, vmax=1, cmap='bone_r')
    im1 = gs[1].imshow(image1, vmin=-diffm, vmax=diffm, cmap='RdBu')
    cb0 = pl.colorbar(im0, cax=gs[0].cax, ticks=[0,1])
    cb1 = pl.colorbar(im1, cax=gs[1].cax, ticks=[-diffm,diffm]) 
    cb0.ax.set_yticklabels(['0', '1'])
    cb1.ax.set_yticklabels(['-%0.1f' % diffm, '%0.1f' % diffm])
    image_names = ["Reference", "Difference"]

    for i in xrange(2):
        gs[i].set_xticks([])
        gs[i].set_yticks([])
        gs[i].set_ylabel(image_names[i])

        if dolabels:
            lbl(gs[i], figlbl[i])

    symbols = ['o', '^', 'D', '>']
    for i in xrange(len(labels)):
        c = COLORS[i]

        if multiple_crbs or i == 0:
            if multiple_crbs:
                label = labels[i] if (i != 0 and not detailed_labels) else '%s CRB' % labels[i]
            else:
                label = 'CRB'
            ax.plot(xs[i], crbs[i], '-', c=c, lw=3, label=label)

        label = labels[i] if (i != 0 and not detailed_labels) else '%s Error' % labels[i]
        ax.plot(xs[i], errors[i], symbols[i], ls='--', lw=2, c=c, label=label, ms=12)

    if dolabels:
        lbl(ax, 'D')
    ax.loglog()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc='upper left', ncol=2, prop={'size': 18}, numpoints=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Position CRB, Error")
    ax.grid(False, which='both', axis='both')
    ax.set_title(title)

    return gs, ax
