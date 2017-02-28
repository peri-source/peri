from __future__ import print_function
from builtins import object

import copy
import numpy as np

class Observer(object):
    def __init__(self):
        pass

    def update(self, state):
        pass

    def reset(self):
        pass

class Printer(object):
    def __init__(self, elems=1, msg=None, skip=1):
        self.elems = elems
        self.msg = msg or "{}, "*(self.elems-1) + '{}'
        self.count = 0
        self.skip = skip

    def update(self, sample):
        if self.count % self.skip == 0:
            if hasattr(sample, '__iter__'):
                msg = self.msg.format(*[i for i in sample])
            else:
                msg = self.msg.format(sample)
            print(self.count, ':', msg)
        self.count += 1

    def reset(self):
        self.count = 0

class TimeAutoCorrelation(Observer):
    def __init__(self):
        self.arr = []

    def update(self, sample):
        self.arr.append(sample)

    def get_correlation(self):
        npn = np.array(self.arr)
        fn = np.fft.fftn(npn)
        return np.real(np.fft.ifftn(fn*fn.conj()))

    def reset(self):
        self.arr = []

class MeanObserver(Observer):
    def __init__(self, block=None):
        self.s = block or np.s_[:]
        self.dat = None
        self.n = 0

    def update(self, sample):
        if self.dat is None:
            self.n = 1
            self.dat = sample[self.s]
        else:
            self.n += 1
            self.dat = self.dat + (sample[self.s] - self.dat) / self.n

    def get_mean(self):
        return self.dat
    
    def reset(self):
        self.dat = None

class CovarianceObserver(Observer):
    def __init__(self, block=None):
        self.s = block or np.s_[:]
        self.mean = None
        self.cov = None
        self.n = 0

    def update(self, sample):
        if self.mean is None:
            self.n = 1
            self.mean = sample[self.s]
            self.cov = 0*np.outer(sample[self.s], sample[self.s])
        else:
            self.n += 1
            self.mean = self.mean + (sample[self.s] - self.mean) / self.n
            self.cov = (self.n-1.0)*self.cov
            self.cov += (self.n-1.0)/self.n*np.outer(sample[self.s]-self.mean,sample[self.s]-self.mean)
            self.cov *= 1.0/self.n

    def get_mean(self):
        return self.mean

    def get_covariance(self):
        return self.cov

    def reset(self):
        self.mean = None

class HistogramObserver(Observer):
    def __init__(self, block=None):
        self.s = block if block is not None else np.s_[:]
        self.dat = []

    def update(self, sample):
        self.dat.append(copy.copy(sample.state[self.s]))

    def get_histogram(self):
        return np.array(self.dat)

    def reset(self):
        self.dat = []
