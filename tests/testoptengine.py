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

    def test_model_cosine(self):
        optfun = make_beale_optfun()
        optfun.update_J()
        model_cosine = optfun.calc_model_cosine()
        self.assertTrue((model_cosine <= 1.0) and (model_cosine >= 0))

    def test_expected_error(self):
        optfun = make_beale_optfun()
        optfun.update_J()
        expected_error = optfun.find_expected_error()
        is_ok = [expected_error > 0, expected_error < optfun.error]
        self.assertTrue(all(is_ok))


class TestStepper(unittest.TestCase):
    def test_constructor(self):
        stepper = make_basic_stepper()
        self.assertTrue(True)

    def test_step(self):
        stepper = make_basic_stepper()
        initial_error = np.copy(stepper.current_error)
        stepper.take_step()
        current_error = stepper.current_error
        self.assertTrue(current_error < initial_error)


def make_beale_optfun():
    function = opttest.beale
    data = np.array([1.5, 2.25, 2.625])
    paramvals = np.zeros(2)
    optfun = optengine.OptFunction(function, data, paramvals)
    return optfun


def make_basic_stepper():
    optfun = make_beale_optfun()
    stepper = optengine.BasicLMStepper(optfun)
    return stepper


if __name__ == '__main__':
    unittest.main()

