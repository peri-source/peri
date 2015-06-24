import numpy as np

def _min(a, b):
    return np.vstack([a, b]).min(axis=0)

def _max(a, b):
    return np.vstack([a, b]).max(axis=0)

class Tile(object):
    def __init__(self, left, right=None, mins=None, maxs=None):
        if right is None:
            right = left
            left = np.zeros(len(left), dtype='int')

        if mins is not None:
            if not hasattr(mins, '__iter__'):
                mins = np.array([mins]*3)
            left = _max(left, mins)

        if maxs is not None:
            if not hasattr(maxs, '__iter__'):
                maxs = np.array([maxs]*3)
            right = _min(right, maxs)

        l, r = left, right
        self.l = np.array(l)
        self.r = np.array(r)
        self.bounds = (l, r)
        self.shape = self.r - self.l
        self.slicer = np.s_[l[0]:r[0], l[1]:r[1], l[2]:r[2]]
