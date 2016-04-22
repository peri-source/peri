import numpy as np
import scipy as sp

from cbamf.util import Tile

def entropy(s):
    t = Tile(s.pad, np.array(s.image.shape)-s.pad)
    m = s.get_model_image()[t.slicer]
    vals = np.linspace(0, 1, 2**8)

    ent = np.zeros_like(m)
    for val in vals:
        prob = 1.0/np.sqrt(2*np.pi*s.sigma**2) * np.exp(-(val-m)**2/(2*s.sigma**2))
        ent += prob * np.log2(prob) * vals[1]
    return ent
