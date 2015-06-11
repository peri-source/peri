import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import acor
import pylab as pl
from colloids.cu import mc, nbody, fields
from colloids.bamf import observers, sampler, model, engine, initialize, state
from time import sleep
import itertools
import sys

GS = 0.05
RADIUS = 9.0

sweeps = 30
samples = 10
burn = sweeps - samples

PAD = int(2*RADIUS)
SIZE = int(3*RADIUS)

#itrue = initialize.load_tiff("/media/scratch/00.tif", do3d=True)[3:,128:256,128:256]
itrue = initialize.load_tiff("/media/scratch/colloids/1_non.tif", do3d=True)[3:,128:256,128:256]
itrue = initialize.normalize(itrue, False)

xstart, imfiltered = initialize.local_max_featuring(itrue, 10)
rstart = 6 + 0*xstart[:,0]
pstart = np.array([1.2,2])

xstart = xstart.astype('double')
rstart = rstart.astype('double')
pstart = pstart.astype('double')

GN = len(rstart)

TPAD = PAD+SIZE/2
impad = np.zeros([itrue.shape[i]+2*TPAD for i in xrange(3)]) - 10
impad[TPAD:-TPAD, TPAD:-TPAD, TPAD:-TPAD] = itrue
itrue = impad
xstart = xstart + TPAD
z,y,x = xstart.T
xstart = np.vstack([x,y,z]).T
strue = np.hstack([xstart.flatten(), rstart, pstart]).copy()

nbody.setSeed(10)
mc.setSeed(10)
np.random.seed(10)

def scan(im, cycles=1):
    pl.figure(1)
    pl.show()
    sleep(3)
    for c in xrange(cycles):
        for sl in im:
            pl.clf()
            pl.imshow(sl, cmap=pl.cm.bone, interpolation='nearest',
                    origin='lower', vmin=0, vmax=1)
            pl.draw()
            sleep(0.03)

def create_tile_per_particle(im, pos, rad, psf, pad, part, size=16):
    """
    Description of confusing parameters -
        * size = the actual image comparison side length. the particle
            should be centered in this cube
        * pad = the amount of padding around that cube in addition

    Therefore, the total cube size of interest if size + 2*pad
    """

    num = part
    if not hasattr(size, '__iter__'):
        size = np.array([size for i in xrange(3)])

    center = np.round(pos[num]).astype('int32')

    x,y,z = pos.T
    pl = center - size/2
    pr = center + size/2
    subim = im[pl[2]:pr[2], pl[1]:pr[1], pl[0]:pr[0]].copy()
    subim[subim > 0] = initialize.normalize(subim[subim > 0])

    mask =  (x > pl[0]-pad) & (x < pr[0]+pad)
    mask &= (y > pl[1]-pad) & (y < pr[1]+pad)
    mask &= (z > pl[2]-pad) & (z < pr[2]+pad)

    nn = np.linspace(0, len(rad)-1, len(rad))
    N = mask.sum()

    xall = pos[mask] - pl
    rall = rad[mask]
    st = np.hstack([xall.flatten(), rall, psf]).astype('float64')
    st = state.StateXRP(N, state=st, partial=True, pad=2*pad, ccd_size=subim.shape[::-1])

    maski =  (xall[:,0]-size[0]/2 > -1) & (xall[:,0]-size[0]/2 < 1)
    maski &= (xall[:,1]-size[1]/2 > -1) & (xall[:,1]-size[1]/2 < 1)
    maski &= (xall[:,2]-size[2]/2 > -1) & (xall[:,2]-size[2]/2 < 1)

    Nmove = maski.sum()
    if Nmove == 0:
        raise IOError

    bpos = (0*st.state[st.b_pos]).astype('bool')
    brad = (0*st.state[st.b_rad]).astype('bool')
    bpsf = (0*st.state[st.b_psf]).astype('bool')
    bpos = bpos.reshape(-1,3)
    bpos[maski] = True
    brad[maski] = True
    bpsf[:] = False

    bs = np.hstack([bpos.flatten(), brad, bpsf])
    blocks = []
    for i in xrange(len(bs)):
        if bs[i]:
            empty = np.zeros(len(bs), dtype='bool')
            empty[i] = True
            blocks.append(empty.copy())

    return blocks, subim, st, pl, mask, N

def sample_state(image, st, blocks, slicing=True):
    m = model.PositionsRadiiPSF(image, imsig=GS)
    bs = blocks

    samplers = [sampler.SliceSampler(RADIUS/1e2, block=b) for b in bs]

    eng = engine.SequentialBlockEngine(m, st)
    opsay = observers.Printer()
    #eng.add_likelihood_observers(opsay)
    eng.add_samplers(samplers)

    error = 0
    print m.has_overlaps(st),
    im0 = m.docalculate(st, False)
    try:
        eng.dosteps(1)
    except Exception as e:
        error = 1
        raise e
    im1 = m.docalculate(st, False)
    print m.has_overlaps(st)

    m.free()
    return im0, im1, st.state.copy(), error

def sample_ll(image, st, element, size=0.1, N=1000):
    m = model.PositionsRadiiPSF(image, imsig=GS)
    start = st.state[element]

    ll = []
    vals = np.linspace(start-size, start+size, N)
    for val in vals:
        st.update(element, val)
        l = m.loglikelihood(st)
        ll.append(l)
    return vals, np.array(ll)

image, x, r, psf, size, = itrue, xstart, rstart, pstart, SIZE
if True:

#def cycle(image, x, r, psf, sweeps=20, burn=10, size=8):
    x, r, psf = x.copy(), r.copy(), psf.copy()

    #type0  = (nbody.ACTIVE+0*r).astype('int32')
    #mc.correct_radii(x.flatten(), r, type0, 0.0005, r)

    h = []
    N0 = r.shape[0]
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        for tile in xrange(N0):

            bl, im, st, offset, mask, N = create_tile_per_particle(image, x, r, psf, PAD, part=tile, size=size)
            ii0, ii1, mu,error = sample_state(im, st, bl, slicing=True)
            st.free()

            old_pos = mu[:3*N]
            old_rad = mu[3*N:4*N]
            mc.naive_renormalize_radii(old_pos, old_rad, 0)

            if error:
                raise IOError
            new_pos = old_pos.reshape(-1,3)
            new_rad = old_rad
            new_psf = mu[4*N:]
            #if error:
            #    raise IOError

            x[mask] = new_pos + offset.reshape(-1,3)
            r[mask] = new_rad
            psf[:] = new_psf

            print tile, (bl[0].shape[0]-2)/4#, '\r',
            sys.stdout.flush()

        if i > burn:
            h.append(np.hstack([x.flatten(), r, psf]))

    h = np.array(h)
    #return h

#h = cycle(itrue, xstart, rstart, pstart, sweeps, sweeps-samples, size=SIZE)
mu = h.mean(axis=0)
std = h.std(axis=0)
pl.figure(figsize=(20,4))
pl.errorbar(xrange(len(mu)), ((mu-strue)/strue), yerr=(std)/np.sqrt(samples),
        fmt='.', lw=0.15, alpha=0.5)
pl.vlines([0,3*GN-0.5, 4*GN-0.5], -1, 1, linestyle='dashed', lw=4, alpha=0.5)
pl.hlines(0, 0, len(mu), linestyle='dashed', lw=5, alpha=0.5)
pl.xlim(0, len(mu))
pl.ylim(-0.02, 0.02)
pl.show()
