import unittest, itertools

import numpy as np; np.random.seed(0)


# for now:
import sys
sys.path.append('../peri/opt')
import optengine, opttest, optimize as opt
import _makestate


TOLS = {'atol': 1e-10, 'rtol': 1e-10}
SOFTTOLS = {'atol': 1e-7, 'rtol': 1e-7}
WEAKTOLS =  {'atol': 1e-6, 'rtol': 1e-6}

STATE = _makestate.make_state()  # I only want to do this once


class TestSeparateParticlesIntoGroups(unittest.TestCase):
    def test_all_in_groups(self):
        unshifted_groups = opt.separate_particles_into_groups(
            STATE, max_mem=1e8)
        unshifted_ok = check_groups(STATE, unshifted_groups)
        shifted_groups = opt.separate_particles_into_groups(
            STATE, max_mem=1e8, doshift=True)
        shifted_ok = check_groups(STATE, shifted_groups)
        self.assertTrue(unshifted_ok)
        self.assertTrue(shifted_ok)


def check_groups(state, groups):
    pg = opt.ParticleGroupCreator(state)
    return pg._check_groups(groups)


if __name__ == '__main__':
    unittest.main()
