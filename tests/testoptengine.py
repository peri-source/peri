import unittest, itertools

import numpy as np; np.random.seed(0)

# for now:
import sys
sys.path.append('../peri/opt')
import optengine, opttest


TOLS = {'atol': 1e-13, 'rtol': 1e-13}
SOFTTOLS = {'atol': 1e-7, 'rtol': 1e-7}
WEAKTOLS =  {'atol': 1e-6, 'rtol': 1e-6}


class TestOptFunction(unittest.TestCase):
    def test_constructor(self):
        optfun = make_optfun('beale')
        self.assertTrue(True)

    def test_error_nonzero(self):
        optfun = make_optfun('beale')
        self.assertTrue(optfun.error > 0)

    def test_update(self):
        optfun = make_optfun('beale')
        optfun.update(np.array([3, 0.5]))
        self.assertTrue(np.isclose(optfun.error, 0, atol=1e-15))

    def test_j(self):
        optfun = make_optfun('beale')
        optfun.update_J()
        self.assertFalse(np.isnan(optfun.J).any())
        self.assertFalse(np.isclose(optfun.J, 0, atol=1e-15).all())

    def test_low_rank_J_update_does_not_drift_params(self):
        params_undrifted = []
        for function_name in opttest.ALL_FUNCTIONS.keys():
            optfun = make_optfun(function_name)
            true_params = optfun.paramvals.copy()
            num_params = true_params.size
            random_direction = np.random.randn(num_params).reshape(1, -1)
            random_direction /= np.linalg.norm(random_direction)

            optfun.low_rank_J_update(random_direction)
            params_undrifted.append(
                np.allclose(true_params, optfun.paramvals, **TOLS))
        self.assertTrue(all(params_undrifted))

    def test_low_rank_J_update_correct_values(self):
        js_correct = []
        params_undrifted = []
        for function_name in opttest.ALL_FUNCTIONS.keys():
            optfun = make_optfun(function_name)
            optfun.update_J()
            true_j = optfun.J.copy()
            true_jtj = optfun.JTJ.copy()

            num_params = optfun.paramvals.size
            # We pick a direction along the coordinate directions,
            # since that should be exactly the same as the J calculation
            # even with roundoff errors:
            direction = np.zeros([1, num_params])
            direction[0, 0] = 1.0

            optfun.low_rank_J_update(direction)
            current_correct = [np.allclose(true_j, optfun.J, **TOLS),
                               np.allclose(true_jtj, optfun.JTJ, **TOLS)]
            js_correct.extend(current_correct)
            if not all(current_correct):
                print(function_name)
        self.assertTrue(all(js_correct))

    def test_model_cosine(self):
        optfun = make_optfun('beale')
        optfun.update_J()
        model_cosine = optfun.calc_model_cosine()
        self.assertTrue((model_cosine <= 1.0) and (model_cosine >= 0))

    def test_expected_error_on_all_linear_models(self):
        all_functions_ok = []
        for optfun in make_all_linear_function_optobjs():
            optfun.update_J()
            current_error = optfun.error
            expected_error = optfun.find_expected_error()
            rescaleby = max([1.0, current_error])
            is_ok = [expected_error <= current_error,
                     np.isclose(expected_error / rescaleby, 0, **TOLS)]
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

    def test_calc_step_does_not_change_error(self):
        both_ok = []
        for stepper in (make_basic_stepper(), make_fancy_stepper()):
            initial_error = np.copy(stepper.optobj.error)
            stepper.optobj.update_J()  # FIXME should this be not-lazy-init?
            step = stepper.calc_simple_LM_step()
            both_ok.append(initial_error == stepper.optobj.error)
        self.assertTrue(all(both_ok))

    def test_accel_step_is_exact_for_linear_models(self):
        errors = []
        all_linear_optfuns = make_all_linear_function_optobjs()
        for optfun in all_linear_optfuns:
            stepper = optengine.FancyLMStepper(optfun, damp=1e-14, accel=True)
            stepper.take_step()
            errors.append(stepper.current_error)
        self.assertTrue(np.allclose(errors, 0, **TOLS))

    def test_accel_step_is_exact_for_rosenbrock_model(self):
        optfun = make_optfun('rosenbrock')
        stepper = optengine.FancyLMStepper(optfun, damp=1e-14, accel=True)
        stepper.take_step()
        self.assertTrue(np.allclose(stepper.current_error, 0, **TOLS))

    def test_broyden_update_does_not_change_J_for_linear_models(self):
        each_ok = []
        for optfun in make_all_linear_function_optobjs():
            stepper = optengine.FancyLMStepper(optfun, damp=1e2, accel=False)
            initial_residuals = optfun.residuals
            initial_paramvals = optfun.paramvals
            stepper.take_step()
            correct_J = stepper.optobj.J.copy()
            correct_JTJ = stepper.optobj.JTJ.copy()
            # We step, get J and JTJ, then broyden update
            direction = optfun.paramvals - initial_paramvals
            delta_residuals = optfun.residuals - initial_residuals
            stepper.broyden_update_J(direction, delta_residuals)
            new_J = stepper.optobj.J.copy()
            new_JTJ = stepper.optobj.JTJ.copy()
            each_ok.extend([np.allclose(correct_J, new_J, **SOFTTOLS),
                            np.allclose(correct_JTJ, new_JTJ, **SOFTTOLS)])
        self.assertTrue(all(each_ok))

    def test_broyden_update_changes_J_for_quadratic_models(self):
        each_ok = []
        for optfun in make_all_quadratic_function_optobjs():
            stepper = optengine.FancyLMStepper(optfun, damp=1e2, accel=False)
            initial_residuals = optfun.residuals
            initial_paramvals = optfun.paramvals
            stepper.take_step()
            correct_J = stepper.optobj.J.copy()
            correct_JTJ = stepper.optobj.JTJ.copy()
            # We step, get J and JTJ, then broyden update
            direction = optfun.paramvals - initial_paramvals
            delta_residuals = optfun.residuals - initial_residuals
            stepper.broyden_update_J(direction, delta_residuals)
            new_J = stepper.optobj.J.copy()
            new_JTJ = stepper.optobj.JTJ.copy()
            each_ok.extend([~np.allclose(correct_J, new_J, **SOFTTOLS),
                            ~np.allclose(correct_JTJ, new_JTJ, **SOFTTOLS)])
        self.assertTrue(all(each_ok))

    def test_eig_update_is_exact_when_complete(self):
        is_ok = []
        for function_name in opttest.ALL_FUNCTIONS.keys():
            optfun = make_optfun(function_name)
            stepper = optengine.FancyLMStepper(optfun)
            optfun.update_J()
            true_j = np.copy(optfun.J)
            true_jtj = np.copy(optfun.JTJ)
            stepper.eig_update_J(number_of_directions=optfun.paramvals.size)
            eig_j = np.copy(optfun.J)
            eig_jtj = np.copy(optfun.JTJ)
            is_ok.append(np.allclose(true_j, eig_j, **WEAKTOLS))
        self.assertTrue(all(is_ok))


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


def make_optfun(key='beale'):
    function_info = opttest.ALL_FUNCTIONS[key]
    optfun = _make_optfun_from_function_info(function_info)
    return optfun


def make_basic_stepper():
    optfun = make_optfun('beale')
    stepper = optengine.BasicLMStepper(optfun)
    return stepper


def make_fancy_stepper():
    optfun = make_optfun('beale')
    stepper = optengine.FancyLMStepper(optfun)
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

