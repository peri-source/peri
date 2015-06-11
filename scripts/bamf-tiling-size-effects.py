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

GN = 128
GS = 0.05
PHI = 0.57
RADIUS = 8.0
PSF = (1.2, 2)

sweeps = 1
samples = 1
burn = sweeps - samples

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
    subim = im[pl[2]:pr[2], pl[1]:pr[1], pl[0]:pr[0]]

    mask =  (x > pl[0]-pad) & (x < pr[0]+pad)
    mask &= (y > pl[1]-pad) & (y < pr[1]+pad)
    mask &= (z > pl[2]-pad) & (z < pr[2]+pad)

    nn = np.linspace(0, len(rad)-1, len(rad))
    N = mask.sum()

    xall = pos[mask] - pl
    rall = rad[mask]
    st = np.hstack([xall.flatten(), rall, psf]).astype('float32')
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

    samplers = [sampler.SliceSampler(RADIUS/1e1, block=b) for b in bs]

    eng = engine.SequentialBlockEngine(m, st)
    opsay = observers.Printer()
    #eng.add_likelihood_observers(opsay)
    eng.add_samplers(samplers)

    error = 0
    im0 = m.docalculate(st, False)
    try:
        eng.dosteps(0)
    except Exception as e:
        error = 1
    im1 = m.docalculate(st, False)

    m.free()
    return im0, im1, st.state.copy(), error

def sample_ll(image, st, element, size=0.1, N=1000):
    m = model.PositionsRadiiPSF(image, imsig=GS)
    start = st.state[element]

    ll = []
    vals = np.linspace(start-size, start+size, N)
    #print element
    for val in vals:
        st.update(element, val)
        l = m.loglikelihood(st)
        #print val, l, st.state[element]
        ll.append(l)
    #return st, m, vals, np.array(ll)
    return vals, np.array(ll)

ys = []
for size in xrange(6, 32):#pref in np.linspace(1,3, 21):
    PAD = int(2*RADIUS)
    SIZE = size#int(pref*RADIUS)

    print SIZE
    import pickle
    #pickle.dump([itrue, xstart, rstart, pstart], open("/media/scratch/bamf_ic.pkl", 'w'))
    itrue, xstart, rstart, pstart = pickle.load(open("/media/scratch/bamf_ic.pkl"))
    #itrue, xstart, rstart, pstart = initialize.fake_image_3d(GN, phi=PHI, noise=GS, radius=RADIUS, psf=PSF)
    TPAD = PAD+SIZE/2
    impad = np.zeros([itrue.shape[i]+2*TPAD for i in xrange(3)]) - 10
    impad[TPAD:-TPAD, TPAD:-TPAD, TPAD:-TPAD] = itrue
    itrue = impad
    xstart = xstart + TPAD
    #rstart = 0.5*rstart
    strue = np.hstack([xstart.flatten(), rstart, pstart]).copy()
    
    nbody.setSeed(10)
    mc.setSeed(10)
    np.random.seed(10)
    
    
    image, x, r, psf, size, = itrue, xstart, rstart, pstart, SIZE
    if True:
        x, r, psf = x.copy(), r.copy(), psf.copy()
    
        h = []
        N0 = r.shape[0]
        for i in xrange(sweeps):
            print '{:=^79}'.format(' Sweep '+str(i)+' ')
    
            for tile in [0]:
    
                bl, im, st, offset, mask, N = create_tile_per_particle(image, x, r, psf, PAD, part=tile, size=size)
                ii0, ii1, mu,error = sample_state(im, st, bl, slicing=True)
    
                new_pos = mu[:3*N].reshape(-1,3)
                new_rad = mu[3*N:4*N]
                new_psf = mu[4*N:]
    
                x[mask] = new_pos + offset.reshape(-1,3)
                r[mask] = np.abs(new_rad)
                psf[:] = new_psf
    
                print tile, (bl[0].shape[0]-2)/4#, '\r',
                sys.stdout.flush()
    
            if i > burn:
                h.append(np.hstack([x.flatten(), r, psf]))
    
        h = np.array(h)
    
    x,y = sample_ll(im, st, bl[-1], size=0.1, N=200)
    ys.append(y-y.max())

pl.imshow(np.array(ys), cmap=pl.cm.bone, origin='lower', aspect='auto')
