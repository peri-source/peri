import unittest

import numpy as np

# for now:
import sys
sys.path.append('../peri/opt')
import optengine, opttest


class TestOptFunction(unittest.TestCase):
    def test_constructor(self):
        optfun = make_beale_optfun()
        self.assertTrue(True)

    def test_error_nonzero(self):
        optfun = make_beale_optfun()
        self.assertTrue(optfun.error > 0)

    def test_update(self):
        optfun = make_beale_optfun()
        optfun.update(np.array([3, 0.5]))
        self.assertTrue(np.isclose(optfun.error, 0, atol=1e-15))

    def test_j(self):
        optfun = make_beale_optfun()
        optfun.update_J()
        self.assertFalse(np.isnan(optfun.J).any())
        self.assertFalse(np.isclose(optfun.J, 0, atol=1e-15).all())


def make_beale_optfun():
    function = opttest.beale
    data = np.array([1.5, 2.25, 2.625])
    paramvals = np.zeros(2)
    optfun = optengine.OptFunction(function, data, paramvals)
    return optfun


if __name__ == '__main__':
    unittest.main()

