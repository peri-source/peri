import numpy as np

from peri import states, util
from peri.test import init, analyze

def get_crb():
    s = init.create_many_particle_state(64, 64, conf=init.conf_cohen_hot)

    xs, ys = [], []
    for i in xrange(1000):
        print i
        x,y = _get_crb(s, np.random.rand(2))
        xs.extend(x)
        ys.extend(y)
    return np.array(xs), np.array(ys)

def _get_crb(s, zy_off=[0,0]):
    
    i = s.obj_closest_particle(s.ishape.shape/2)
    p = s.param_particle_pos(i)

    # set the other coordinates
    x0 = np.floor(s.get_values(p))
    x0 += np.array([zy_off[0], zy_off[1], 0.0])
    s.update(p, x0)

    # now we're only talking about the x-coordinate
    p = p[-1]
    x0 = x0[-1]
    
    xs, ys = [], []
    for x in np.linspace(x0, x0+1, 20):
        s.update(p, x)
    
        xs.append(x)
        ys.append(s.crb(p))
    
    xs = np.array(xs).squeeze() - x0
    ys = np.array(ys).squeeze()

    return xs, ys

def sim():
    p = np.linspace(0, 1, int(1e8))
    c = lambda x: 0.025 + 0.013*(1-np.cos(2*np.pi*x))
    s = c(p)*np.random.randn(len(p)) + p
    return s

def get_crb_exp(files):

    xs = []
    for f in files:
        x = np.loadtxt(f)[:,2]
        xs.extend(x)
    return np.array(xs)


