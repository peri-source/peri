import numpy as np

class Tile(object):
    def __init__(self, left, right=None):
        if right is None:
            right = left
            left = np.zeros(len(left), dtype='int')

        l, r = left, right
        self.l = np.array(l)
        self.r = np.array(r)
        self.bounds = (l, r)
        self.shape = self.r - self.l
        self.slicer = np.s_[l[0]:r[0], l[1]:r[1], l[2]:r[2]]
