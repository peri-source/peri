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
        unshifted_ok = check_all_particles_in_groups(STATE, unshifted_groups)
        shifted_groups = opt.separate_particles_into_groups(
            STATE, max_mem=1e8, doshift=True)
        shifted_ok = check_all_particles_in_groups(STATE, shifted_groups)
        self.assertTrue(unshifted_ok)
        self.assertTrue(shifted_ok)

    def test_groups_not_too_large(self):
        for doshift in [True, False]:
            groups = opt.separate_particles_into_groups(
                STATE, max_mem=1e8, doshift=doshift)
            size_ok = check_groups_not_too_large(STATE, groups, max_mem=1e8)
            self.assertTrue(size_ok)

    def test_split_group_creator_init(self):
        pgc = opt.ListParticleGroupCreator(STATE, max_mem=1e8)
        self.assertTrue(True)

    def test_split_group_creator_all_in_groups(self):
        for split_by in [2, 3]:
            pgc = opt.ListParticleGroupCreator(
                STATE, max_mem=1e8, split_by=split_by)
            groups = pgc.separate_particles_into_groups()
            self.assertTrue(check_all_particles_in_groups(STATE, groups))

    def test_split_groups_not_too_large(self):
        for split_by in [2, 3]:
            pgc = opt.ListParticleGroupCreator(
                STATE, max_mem=1e8, split_by=split_by)
            groups = pgc.separate_particles_into_groups()
            size_ok = check_groups_not_too_large(STATE, groups, max_mem=1e8)
            self.assertTrue(size_ok)


def check_all_particles_in_groups(state, groups):
    pg = opt.ParticleGroupCreator(state)
    return pg._check_groups(groups)


def check_groups_not_too_large(state, groups, max_mem=1e8):
    pg = opt.ParticleGroupCreator(state, max_mem=1e8)
    sizes_ok = [pg.calc_group_memory_bytes(g) < max_mem for g in groups]
    return all(sizes_ok)


if __name__ == '__main__':
    unittest.main()
