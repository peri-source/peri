import unittest, itertools

import numpy as np

# for now:
import sys
sys.path.append('../peri/opt')
import optengine, opttest

# Unittests to add:
# 1. LM steps to exactly correct parameters for linear models when damping
#    is 0 (basic + fancy).
# 2. Same as above for linear + quadratic models with use_accel

TOLS = {'atol': 1e-11, 'rtol': 1e-11}


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

    def test_expected_error_on_all_linear_models(self):
        all_functions_ok = []
        for optfun in make_all_linear_function_optobjs():
            optfun.update_J()
            expected_error = optfun.find_expected_error()
            is_ok = [expected_error <= optfun.error,
                     np.isclose(expected_error, 0, **TOLS)]
            all_functions_ok.append(all(is_ok))
        self.assertTrue(all(all_functions_ok))

    def test_model_cosine_on_all_quadratic_models(self):
        all_functions_ok = []
        for optfun in make_all_linear_function_optobjs():
            optfun.update_J()
            model_cosine = optfun.calc_model_cosine()
            is_ok = np.isclose(model_cosine, 1, **TOLS)
            all_functions_ok.append(is_ok)
        self.assertTrue(all(all_functions_ok))


class TestStepper(unittest.TestCase):
    def test_constructor(self):
        stepper = make_basic_stepper()
        self.assertTrue(True)

    def test_step_decreases_error(self):
        stepper = make_basic_stepper()
        initial_error = np.copy(stepper.current_error)
        stepper.take_step()
        current_error = stepper.current_error
        self.assertTrue(current_error <= initial_error)

    def test_lm_step_is_exact_for_linear_models(self):
        errors_near_0 = []
        for optfun in make_all_linear_function_optobjs():
            stepper = optengine.BasicLMStepper(optfun, damp=1e-14)
            stepper.take_step()
            errors_near_0.append(np.isclose(stepper.current_error, 0, **TOLS))
        self.assertTrue(all(errors_near_0))


class TestOptimizer(unittest.TestCase):
    def test_optimize(self):
        optimizer = make_optimizer()
        optobj = optimizer.stepper.optobj
        optimizer.optimize()
        error_converged = np.isclose(optobj.error, 0, atol=1e-9)
        params_converged = np.allclose(optobj.paramvals,
                                      np.array([3, 0.5]), atol=1e-9)
        self.assertTrue(all([error_converged, params_converged]))


def make_all_linear_function_optobjs():
    for function_info in opttest.LINEAR_FUNCTIONS.values():
        optfun = _make_optfun_from_function_info(function_info)
        yield optfun


def make_all_quadratic_function_optobjs():
    for function_info in opttest.QUADRATIC_FUNCTIONS.values():
        optfun = _make_optfun_from_function_info(function_info)
        yield optfun


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


def make_optimizer():
    stepper = make_basic_stepper()
    optimizer = optengine.Optimizer(stepper)
    return optimizer


def _make_optfun_from_function_info(function_info):
    function = function_info['function']
    data = function_info['data']
    initial_param_guess = 0 * function_info['true-params']
    optfun = optengine.OptFunction(function, data, initial_param_guess)
    return optfun


if __name__ == '__main__':
    unittest.main()

