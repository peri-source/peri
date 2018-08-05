# TODO: optengine only (ignoring changes that depend on states.py for now)
"""
    1. Decimation in OptState. Would calculate which indices to sample.
        -- this could stay in optimize.py; not sure
        -- in decimate_state for now; un touched.
    2. OptimizationScheme / GroupedOptimizer.
        a. Do line / quadratic / etc minimization.
"""
# TODO: updates that depend on states.py
"""
    1. Update st.gradmodel_e to st.gradmodel_error
    2. residuals = model - data (so gradmodel is gradresiduals); this
       might be already done in which case wtf?
"""

import numpy as np
import time
from peri.logger import log
CLOG = log.getChild('optengine')


def _low_mem_mtm(m, step='calc'):
    """np.dot(m.T, m) with low mem usage for m C-ordered, by using small steps

    Parameters
    ----------
    m : numpy.ndarray
        The matrix whose transpose to dot
    step : Int or `'calc'`, optional
        The size of the chunks to do the dot product in. Defualt is 1%
        additional mem overhead.

    Returns
    -------
    mtm : numpy.ndarray
        Array equal to np.dot(m.T, m)
    """
    if not m.flags.c_contiguous:
        raise ValueError('m must be C ordered for this to work with less mem.')
    if step == 'calc':
        step = np.ceil(1e-2 * m.shape[0]).astype('int')
    mtm = np.zeros([m.shape[1], m.shape[1]])  # 6us
    for a in range(0, m.shape[1], step):
        max_index = min(a+step, m.shape[0])
        mtm[a:max_index, :] = np.dot(m[:, a:max_index].T, m)
    return mtm


def get_residuals_update_tile(st, params, values=None):
    """Finds the update tile of a state in the residuals (unpadded) image.

    Parameters
    ----------
    st : :class:`peri.states.State`
        The state
    params : list of valid parameter names
        The parameter names to find the update tile for.
    values : list of valid param values or None, optional
        The update values of the parameters. If None, uses their
        current values.

    Returns
    -------
    :class:`peri.util.Tile`
        The tile corresponding to padded_tile in the unpadded image.
    """
    if values is None:
        values = st.get_values(params)
    itile = st.get_update_io_tiles(params, values)[1]
    inner_tile = st.ishape.intersection([st.ishape, itile])
    return inner_tile.translate(-st.pad)


def decimate_state(state, nparams, decimate=1, min_redundant=20, max_mem=1e9):
    """
    Given a state, returns set of indices """
    pass  # FIXME

# The Engine should be flexible enough where it doesn't care whether residuals
# is data-model or model-data, since the OptObj just provides residuals.

# OPTOBJ and daughter classes:
# TODO:
"""
    1. An exact _graderr for the OptState.
    2. A graderr which uses the exact _graderr when it should and a JT*r
       when it can't use the exact graderr
"""

# ISDONE:
"""
    1. raise error if J has nans
    2. Calculate the gradient of the cost and the residuals, with low memory
    3. Does the optobj know about JTJ too? certainly not a damped JTJ though.
    4. Low-rank J updates
    5. Consistent notation for J -- J[i,j] = ith residual, j param
    6. In OptObj's J is not initialized with the init, but is created
       once when needed and then re-filled afterwards. So there is only
       one memory allocate call, and it is not done before it is needed.
    7. Low-mem JTJ, J*r
    8. rts=True option (rts = the kwargs to return-to-start at each
       point in the J update)
"""


class OptObj(object):
    def __init__(self, param_ranges=None):
        """Superclass for optimizing an object, especially for minimizing
        the sum of the squares of a residuals vector.
        This object knows everything that might be needed about the fit
        landscape, including:
        * The residuals vector
        * The current cost (`error`), not necessarily the sum of the squares
          of the residuals vector (e.g. decimated residuals but accurate cost
          or priors)
        * The gradient of the residuals vector, and a way to update it when
          desired
        * The gradient of the cost, and a way to update it with function
          updates / when desired.
        * The current parameter values
        * How to update the object to a new set of parameter values
        It knows nothing about methods to compute steps for those objects,
        however.
        """
        self.param_ranges = (np.array(param_ranges) if param_ranges is not None
                             else param_ranges)

    def clean_mem(self):
        """Cleans up any large blocks of memory used by the object"""
        self.J = None
        self.JTJ = None

    def update_J(self):
        """Updates the Jacobian / gradient of the residuals = gradient of model
        Ideally stored as a C-ordered numpy.ndarray
        """
        raise NotImplementedError("Implement in subclass")

    def update(self, values):
        """Updates the function to `values`, clipped to the allowed range"""
        if self.param_ranges is not None:
            goodvals = np.clip(values, self.param_ranges[:, 0],
                               self.param_ranges[:, 1])
        else:
            goodvals = values
        return self._update(values)

    def _update(self, values):
        """Update the object; returns the error"""
        raise NotImplementedError("Implement in subclass")

    def gradcost(self):
        """Returns the gradient of the cost"""
        return 2*np.dot(self.residuals, self.J)

    @property
    def error(self):
        r = self.residuals.copy()
        return np.dot(r, r)

    @property
    def residuals(self):
        """Returns the residuals = model-data"""
        raise NotImplementedError('Implement in subclass')

    @property
    def paramvals(self):
        """Returns the current value of the parameters"""
        raise NotImplementedError('Implement in subclass')

    def low_rank_J_update(self, direction, values=None):
        """Does a series of rank-1 updates on self.J.

        direction : numpy.ndarray
            [n, m] shaped array of the (n) directions, each with m elements
            corresponding to the parameter update direction. Must be normalized
            (m = self.paramvals.size)
        values : numpy.ndarray or None, optional
            If not None, the corresponding values of J along each of the n
            directions -- i.e a [n, M] element array with each of the n
            elements corresponding to the M-element tangent vector in data
            space [n=# of grad directions, M=self.residuals.size]. If None,
            computed automatically

        Does this by doing J += np.outer(direction, new_values - old_values)
        without using lots of memory
        """
        hasvals = values is not None
        if hasvals:
            if np.shape(direction)[0] != np.shape(values)[0]:
                raise ValueError('direction, value must be same shape')
        else:
            r0 = self.residuals.copy()
            p0 = self.paramvals.copy()
        if self.J is None:
            self._initialize_J()
        for a, d in enumerate(direction):
            if not np.isclose(np.dot(d, d), 1.0, atol=1e-10):
                raise ValueError('direction is not normalized')
            vals_to_sub = np.dot(self.J, d)
            if hasvals:
                vals = values[a]
            else:
                dp = d * self.dl
                self.update(p0+dp)
                r1 = self.residuals.copy()
                vals = (r1-r0) / self.dl
            delta_vals = vals - vals_to_sub
            for b in range(d.size):
                self.J[:, b] += d[b] * delta_vals
        if np.isnan(self.J.sum()):
            raise RuntimeError('J has nans')
        self._calcjtj()

    def _calcjtj(self):
        self.JTJ = _low_mem_mtm(self.J)
        if np.isnan(self.JTJ.sum()):
            raise RuntimeError('JTJ has nans')

    def find_expected_error(self, delta_params='perfect'):
        """Returns the error expected after an update if the model were linear.

        Parameters
        ----------
        delta_params : {numpy.ndarray or 'perfect'}, optional
            The relative change in parameters. If `'perfect'`, uses the
            best parameters for a perfect linear fit.

        Returns
        -------
        numpy.float64
            The expected error after the update with `'delta_params'`
        """
        grad = self.gradcost()
        if delta_params == 'perfect':
            delta_params = np.linalg.lstsq(self.JTJ, -0.5*grad, rcond=1e-13)[0]
        # If the model were linear, then the residuals would be
        # r = r0 + theta * J
        # The cost would be quadratic in theta; c = (r0 + theta * J)^2
        # The gradient would be grad = 2 * J * (r0 + theta * J),
        # and the Hessian would be 2 * JTJ
        # with Hessian 2*`self.JTJ` and gradient `grad`
        expected_error = (self.error + np.dot(grad, delta_params) +
                          np.dot(np.dot(self.JTJ, delta_params), delta_params))
        return expected_error

    def calc_model_cosine(self):
        """Calculates the cosine of the residuals with the model, based
        on the expected error of the model.

        Returns
        -------
        abs_cos : numpy.float64
            The absolute value of the model cosine.

        Notes
        -----
        The model cosine is defined in terms of the geometric view of
        curve-fitting, as a model manifold embedded in a high-dimensional
        space. The model cosine is the cosine of the residuals vector
        with its projection on the tangent space:
            :math:`cos(phi) = |P^T r|/|r|`
        where :math:`P^T` is the projection operator onto the model
        manifold and :math:`r` the residuals.

        Rather than doing the SVD, we get this from the expected error
        of the model if it were linear. We use that expected error and
        the current error to calculate a model sine, and use that to
        get a model cosine (this is faster than SVD and better-conditioned).
        """
        expected_error = self.find_expected_error(delta_params='perfect')
        model_sine_2 = expected_error / (self.error + 1e-9)  # error=distance^2
        abs_cos = np.sqrt(1 - model_sine_2)
        return abs_cos


class OptFunction(OptObj):
    def __init__(self, function, data, paramvals, dl=1e-7, **kwargs):
        """OptObj for a function

        Parameters
        ----------
        function : callable
            Must return a np.ndarray-like with syntax function(data, *params)
            and return an object of the same size as ``data``
        data : flattened numpy.ndarray
            Passed to the function. Must return a valid ``np.size``
        paramvals : numpy.ndarray
            extra args to pass to the function.
        dl : Float, optional
            Step size for finite-difference (2pt stencil) derivative
            approximation. Default is 1e-7
        param_ranges : [N, 2] list-like or None, optional
            If not None, valid parameter ranges for each of the N
            parameters, with `param_ranges[i]` being the [low, high]
            bounding values. Default is None, for no bounds on the
            parameters.
        """
        self.function = function
        self.data = data
        self._model = np.zeros_like(data)
        self._paramvals = np.array(paramvals).reshape(-1)
        self.J = None  # we don't create J until we need to
        self.dl = dl
        super(OptFunction, self).__init__(**kwargs)

    def _initialize_J(self):
        self.J = np.zeros([self.data.size, self.paramvals.size], dtype='float')

    def update_J(self):
        if self.J is None:
            self._initialize_J()
        r0 = self.residuals.copy()
        p0 = self._paramvals.copy()
        dp = np.zeros_like(p0)
        for i in range(p0.size):
            dp *= 0
            dp[i] = self.dl
            self.update(p0 + dp)
            r1 = self.residuals.copy()
            self.J[:, i] = (r1-r0) / self.dl
        if np.isnan(self.J.sum()):
            raise RuntimeError('J has nans')
        self._calcjtj()
        # And we put params back:
        self.update(p0)

    def _update(self, values):
        self._paramvals[:] = values
        self._model = self.function(self._paramvals)
        return self.error

    @property
    def residuals(self):
        return np.ravel(self._model - self.data)

    @property
    def paramvals(self):
        return self._paramvals.copy()


# Only works for image state because the residuals update tile only works
# for image states -- get_update_io_tile is only for ImageState
class OptImageState(OptObj):
    def __init__(self, state, params, dl=3e-6, inds=None, rts=True, **kwargs):
        """OptObj for a peri.states.ImageState

        Parameters
        ----------
        state : `peri.states.ImageState`
            The state to optimize
        params : list
            The parameters of the state to optimize
        dl : Float, optional
            Step size for finite-difference approximation of model
            derivates. Default is 3e-6
        inds : numpy.ndarray, slice, or None, optional
            Indices of the (raveled) residuals to use for calculating
            steps. Used to save memory. Default is None
        rts : Bool, optional
            Whether or not to return the state to its original value
            before evaluating the derivative of each parameter. Set
            to True for an accurate J calculation, False for a fast
            one, as passed to states.gradmodel. Default is True.
        param_ranges : [N, 2] list-like or None, optional
            If not None, valid parameter ranges for each of the N
            parameters, with `param_ranges[i]` being the [low, high]
            bounding values. Default is None, for no bounds on the
            parameters.
        """
        self.state = state
        self.params = params
        self.dl = dl
        self.rts = rts
        # self.tile = get_residuals_update_tile(state, params)
        # if inds is None:
        #     self.inds = slice(None)
        # else:
        #     self.inds = inds
        # FIXME this is ugly. Could just be the nice thing above
        # if states.sample took both a `inds` and a `tile`
        # as far as I can tell states.sample is only used for gradmodel
        # really, although it is also used (through st.m, st.r) for:
        #       * st.fisherinformation
        #       * st.gradloglikelihood
        #       * st.hessloglikelihood
        #       * st.gradmodel
        #       * st.hessmodel
        #       * st.JTJ
        #       * st.J
        #       * st.J_e
        #       * st.gradmodel_e
        # -- note that there is a states.J, so you can use st.J =
        #    st.gradresiduals
        tile = get_residuals_update_tile(state, params)
        if tile == state.ishape.translate(-state.pad):
            # full residuals, no tile:
            self._slicer = None
        else:
            self._slicer = tile.slicer
        if self._slicer is None:
            self.inds = slice(None) if inds is None else inds
        else:  # _slicer is not None:
            if inds is not None:
                raise ValueError("Currently cannot set both inds and slicer.")
            self.inds = inds
        # end fixme
        self.J = None
        super(OptImageState, self).__init__(**kwargs)

    def _initialize_J(self):
        self.J = np.zeros([self.residuals.size, len(self.params)])

    def update_J(self):
        # FIXME s :
        # (1) would be nice if you could pre-allocate J in state.gradmodel
        # self._initialize_J() -- not yet because the gradmodel is backwards
        # (2) would be nice if gradmodel took both slicer and inds, slicer
        #     first then inds.
        # (3) gradmodel J order needs to be swapped
        if self.J is None:
            self._initialize_J()
        start = time.time()
        # self.J[:] = np.transpose(self.state.gradmodel(
        #     params=self.params, rts=self.rts, dl=self.dl,
        #     slicer=self.tile.slicer, inds=self.inds))
        self.J[:] = np.transpose(self.state.gradmodel(
            params=self.params, rts=self.rts, dl=self.dl,
            slicer=self._slicer, inds=self.inds))  # FIXME ugly, prettier above
        CLOG.debug('Calculated J:\t{:.1f} s'.format(time.time() - start))
        self._calcjtj()

    @property
    def residuals(self):
        # FIXME residuals = model - data
        # return -self.state.residuals[self.tile.slicer].ravel()[self.inds]
        # FIXME ugly, prettier above
        if self.inds is not None:
            return -self.state.residuals.ravel()[self.inds]
        elif self._slicer is not None:
            return -self.state.residuals[self._slicer].ravel()
        else:
            raise RuntimeError('wtf')

    @property
    def error(self):
        return self.state.error

    @property
    def paramvals(self):
        return np.copy(self.state.get_values(self.params))

    def _update(self, values):
        self.state.update(self.params, values)
        return self.error


# LMOPTIMIZER (there should be no daughter classes!!!!)
# TODO
"""
    0. Keep it...
        a. only relying on OptObj
        b. agnostic to whether residuals is data-model or model-data
    1. is-a rather than has-a attr/class relation. It seems like the
       optimize should "have a" additional step, damper mode, etc
            - damping
            - step calculation with acceleration
            - take-additional-steps mode (e.g. run with J vs sub-
              block runs vs w/e) -- should I remove subblock runs?
    2. Test it when finished to ensure that it works nicely
    3. Multiple "take additional step" modes....
    4. Possible new features:
        a. "Guess updates when stuck / failed step?"
            - I have no idea what that comment means
        b. A better way to update the damping significantly when needed.
            - honestly I think this need will disappear now that the
              rts=True vs rts=False issue is known from the biases paper
"""

# ISDONE
"""
    1. If it has a J, it needs to know how to do partial updates of J.
       - for instance, it needs to know how to do an eigen update of J
       and a Broyden update of J. And then it needs to know how to
       intelligently update them (i.e. only Broyden update when we take
       a good step, not when we try out to infinity on accident)
    2. Clear termination criterion
      a. Terminate when stuck.
      b. Returned termination flags (e.g. completed, stuck, maxiter)
    3. Only 1 run mode.
    4. Separate increase/decrease damping factors.
    5. geodesic acceleration
    6. Internal run
    7. Broyden updates
    8. Ability to pick multiple damping modes or supply a user-defined
       mode.
"""

# damping functions:
def damp_additive(jtj, damp):
    return jtj + np.eye(jtj.shape[0]) * damp


def damp_multiplicative(jtj, damp):
    return jtj + np.diag(np.diag(jtj)) * damp


def damp_cutoff(jtj, damp, minval=1.0):
    """Damping matrix proportional to diag(jtj), with a minimum value"""
    return jtj + np.diag(np.clip(np.diag(jtj), minval, np.inf))


# one other which are not so simple because it is history dependent:
# 1. Pick damping matrix to be diagonal, with entries the largest
#    diagonal entries of JTJ yet encountered. (Minpack)
# (could also add some experimental ones, but I don't see the point)

DAMPMODES = {'additive': damp_additive,
             'multiplicative': damp_multiplicative,
             'cutoff': damp_cutoff}


class Optimizer(object):
    def __init__(self, stepper, exptol=1e-7, costol=1e-5, errtol=1e-7,
                 fractol=1e-7, paramtol=1e-7, maxiter=100, do_clean_mem=True):
        self.stepper = stepper
        self.exptol = exptol
        self.costol = costol
        self.errtol = errtol
        self.fractol = fractol
        self.paramtol = paramtol
        self.maxiter = maxiter
        self.do_clean_mem = do_clean_mem

    def check_is_completed(self):
        """Return a bool of whether or not optimization has converged"""
        convergence_stats = self.get_convergence_stats()
        has_converged = any([v < getattr(self, k)
                             for k, v in convergence_stats.items()])
        return has_converged

    def get_convergence_stats(self):
        """Returns a dict of termination info"""
        stepper = self.stepper
        d = {
            'errtol': stepper.change_in_error,
            'fractol': stepper.change_in_error / stepper.current_error,
            'paramtol': np.abs(stepper.change_in_parameters).max(),
            'costol': stepper.evaluate_model_cosine(),
            'exptol': stepper.current_error - stepper.find_expected_error(),
            }
        return d

    def optimize(self):
        for _ in range(self.maxiter):
            self.stepper.take_step()
            if self.check_is_completed():
                flag = 'converged'
                break
        else:
            flag = 'unconverged'
        if self.do_clean_mem:
            self.stepper.clean_mem()
        return flag


# TODO: add log debugs on steps
class Stepper(object):
    def __init__(self, optobj):
        self.optobj = optobj

    def take_step(self):  # FIXME better name
        initial_paramvals = np.copy(self.optobj.paramvals)
        initial_error = np.copy(self.current_error)
        self.execute_one_optimization_step()
        self.change_in_error = initial_error - self.current_error
        self.change_in_parameters = initial_paramvals - self.optobj.paramvals

    def execute_one_optimization_step(self):  # FIXME better name
        """Actually takes the steps. Calculates the paramvals to evaluate
        and updates the optobj however many times it needs to.

        The actual bookkeeping is done in take_step.
        """
        raise NotImplementedError('Implement in subclass!')

    def evaluate_model_cosine(self):
        return self.optobj.calc_model_cosine()

    def find_expected_error(self):
        return self.optobj.find_expected_error()

    @property
    def current_error(self):
        return self.optobj.error

    def clean_mem(self):
        # Could do more in subclasses
        self.optobj.clean_mem()


class BasicLMStepper(Stepper):
    def __init__(self, optobj, damp=1.0, dampdown=8.0, dampup=3.0,
                 dampmode='additive'):
        super(BasicLMStepper, self).__init__(optobj)
        self.damp = damp
        self.dampdown = dampdown
        self.dampup = dampup
        self.dampmode = dampmode
        self._rcond = 1e-15  # rcond for np.linalg.lstsq on matrix solution

    def try_step(self, step):
        """Try step. If error decreases, decrease damping. If it increases,
        increase damping and go back."""
        initial_error = np.copy(self.optobj.error)
        initial_paramvals = np.copy(self.optobj.paramvals)
        self.optobj.update(initial_paramvals + step)
        step_is_ok = self.optobj.error < initial_error
        if step_is_ok:
            self.decrease_damping()
        else:
            self.optobj.update(initial_paramvals)
            self.increase_damping()
        return step_is_ok

    def execute_one_optimization_step(self):
        self.optobj.update_J()
        step = self.calc_simple_LM_step()
        self.try_step(step)

    def calc_simple_LM_step(self):
        dampedJTJ = self.damp_JTJ()
        gradcost = self.optobj.gradcost()
        return np.linalg.lstsq(dampedJTJ, -0.5*gradcost, rcond=self._rcond)[0]

    def decrease_damping(self):
        # splitting this out for polymorphism
        self.damp /= self.dampdown

    def increase_damping(self):
        # splitting this out for polymorphism
        self.damp *= self.dampup

    def damp_JTJ(self):
        # splitting this out for polymorphism
        dampedJTJ = (self.optobj.JTJ +
                     self.damp * np.eye(self.optobj.JTJ.shape[0]))
        return dampedJTJ


# TODO implement all the bells and whistles you used in this stepper
class FancyLMStepper(BasicLMStepper):
    def __init__(self, optobj, damp=1.0, dampdown=8., dampup=3., dampmode=
                 'additive', nsteps=(1, 2), accel=True, dobroyden=True,
                 eigupdate=True):
        super(FancyLMStepper, self).__init__(
            optobj, damp=damp, dampdown=dampdown, dampup=dampup,
            dampmode=dampmode)
        self.nsteps = nsteps
        self.accel = accel
        self.dobroyden = dobroyden
        self.eigupdate = eigupdate

    def execute_one_optimization_step(self):
        self.optobj.update_J()
        simple_step = self.calc_simple_LM_step()
        accel_correction = self.calc_accel_correction(simple_step)
        step = simple_step + accel_correction
        step_is_ok = self.try_step(step)
        # try_step would get the broyden and eig updates (here only)
        # whereas run with J etc would follow here.

    def calc_accel_correction(self, initialstep):
        """Calculates geodesic acceleration correction, exact for
        quadratic models."""
        # 1. Get the second derivative of the residuals along the
        #    direction of initialstep
        p0 = self.optobj.paramvals.copy()

        self.optobj.update(p0)
        residuals_at_p0 = self.optobj.residuals.copy()

        self.optobj.update(p0 + initialstep)
        residuals_at_plus = self.optobj.residuals.copy()

        self.optobj.update(p0 - initialstep)
        residuals_at_minus = self.optobj.residuals.copy()

        self.optobj.update(p0)

        second_derivative = (residuals_at_plus
                             - 2 * residuals_at_p0
                             + residuals_at_minus)

        # 2. Calculate the acceleration correction
        dampedJTJ = self.damp_JTJ()
        correction = np.linalg.lstsq(
            dampedJTJ, np.dot(second_derivative, self.optobj.J),
            rcond=self._rcond)[0]
        correction *= -0.5
        return correction

    def broyden_update_J(self, direction, delta_residuals):
        """Update J along `direction` for change in residuals `dr` (both 1D)"""
        CLOG.debug('Broyden update.')
        direction_magnitude = np.linalg.norm(direction)
        if direction_magnitude > 0:  # edge case when at the  exact minimum
            normalized_direction = direction / direction_magnitude
            normalized_delta_residuals = delta_residuals / direction_magnitude
            self.optobj.low_rank_J_update(
                normalized_direction.reshape(1, -1),
                normalized_delta_residuals.reshape(1, -1))


class LMOptimizer(object):
    def __init__(self, optobj, damp=1.0, dampdown=8., dampup=3., dampmode=
                 'additive', nsteps=(1, 2), accel=True, dobroyden=True,
                 exptol=1e-7, costol=1e-5, errtol=1e-7, fractol=1e-7,
                 paramtol=1e-7, maxiter=2, clean_mem=True):
        """Optimizes an OptObj using the Levenberg-Marquardt algorithm.

        Parameters
        ----------
        optobj : OptObj instance
            The OptObj to optimize.
        damp : float, optional
        dampdown, dampup : float > 1, optional
            The amount to decrease or increase the damping by when after
            a successful or failed step, respectively. Defaults are 8
            and 3
        dampmode : {'additive', 'multiplicative', 'cutoff'}, optional
        nsteps : tuple, optional
        accel : bool, optional
        dobroyden : bool, optional
        exptol : float, optional
        costol : float, optional
        errtol : float, optional
        fractol : float, optional
        paramtol : float, optional
        maxiter : int, optional
        clean_mem : float, optional

        Methods
        -------
        optimize()
            Runs the optimization
        check_completion()
            Return a bool of whether or not optimization has converged
        get_convergence_stats()
            Returns a dict of termination info

        Notes
        -----
        This engine knows _nothing_ about the current state of the object --
        e.g. the current cost, current residuals, gradient of the error, etc
        -- all of this is known only by the optobject.
        The optimizer only knows how to take steps given that information.
        """
        self.optobj = optobj
        self.lastvals = self.optobj.paramvals.copy()
        self._rcond = 1e-13  # rcond for leastsq step, after damping

        self.damp = damp
        self.dampdown = dampdown
        self.dampup = dampup
        self.nsteps = nsteps
        if dampmode in DAMPMODES.keys():
            self.dampfunc = DAMPMODES[dampmode]
            self.dampmode = dampmode
        else:
            if not hasattr(dampmode, '__call__'):
                raise ValueError(
                    "`dampmode` must either be one of {} or callable.".format(
                    DAMPMODES.keys()))
            self.dampfunc = dampmode  # could ensure that is is callable
            self.dampmode = 'custom'
        self.accel = accel
        self.dobroyden = dobroyden

        # Termination dict:
        self.maxiter = maxiter
        self.term_dict = {'errtol': errtol, 'fractol': fractol,
                          'paramtol': paramtol, 'exptol': exptol,
                          'costol': costol}
        self.clean_mem = clean_mem

    def take_initial_step(self):
        """Takes a step after a fresh J and updates the damping"""
        # 2. Calculate & take a step -- distinction might be blurred
        #       a. Calculate N steps
        #               i.  The damping will be updated during this process
        #               ii. Might involve acceleration etc steps
        #       b. Pick the best step that is also good
        #               i.  If good, take that step
        #               ii. If not good, increase damping somehow til step
        #                   is good
        obj = self.optobj
        lasterror = obj.error * 1.0
        lastvals = obj.paramvals.copy()
        lastresiduals = obj.residuals.copy()
        damps = [self.damp / (self.dampdown**i) for i in range(*self.nsteps)]
        steps = [self.calc_step(self.damp_JTJ(d)) for d in damps]
        errs = [obj.update(lastvals + step) for step in steps]
        best = np.nanargmin(errs)
        CLOG.debug('Initial Step:')
        CLOG.debug('\t'.join(['{:.3f}'.format(e) for e in [lasterror] + errs]))
        if errs[best] < lasterror:  # if we found a good step, take it
            CLOG.debug('Good step')
            self.damp = damps[best]
            # possibly 1 wasted func eval below FIXME:
            er = obj.update(lastvals + steps[best])
            test = np.abs(1-(er+1e-16)/(errs[best]+1e-16)) < 1e-10
            msg = 'Inexact updates'
            assert test, msg
        else:
            # Increase damping until good step
            CLOG.debug('Bad step, increasing damping')
            for _ in range(15):
                self.damp *= self.dampup
                step = self.calc_step(self.damp_JTJ(self.damp))
                err = obj.update(lastvals + step)
                if err < lasterror:
                    CLOG.debug('Increased damping {}x, {:.3f}'.format(_, err))
                    break
            else:  # for-break-else, failed to increase damping
                # update function to previous value, terminate?
                obj.update(lastvals)
                CLOG.warn('Stuck!')
                return 'stuck'
        # If we're still here, we've taken a good step, so broyden update:
        CLOG.debug('Initial step: \t{:.3f}'.format(obj.error))
        self.lasterror = lasterror
        self.lastvals[:] = lastvals.copy()
        if self.dobroyden:
            self.broyden_update_J(obj.paramvals - lastvals,
                                  obj.residuals - lastresiduals)
        return 'run'

    def take_additional_steps(self):
        """Takes additional steps"""
        # Right now just run with J, but could be a more complicated procedure.
        return self.run_with_J()

    def damp_JTJ(self, damp):
        # One possible different option is to pass a self.dampfunc as a
        # function which takes JTJ, damping and returns a damped JTJ
        # -- right now defined explicitly in the init
        return self.dampfunc(self.optobj.JTJ, damp)

    # Various internal run modes:
    def run_with_J(self, maxrun=6):
        """Takes up to `maxrun` steps without updating J or the damping"""
        obj = self.optobj
        for _ in range(maxrun):
            lasterror = 1.0 * obj.error
            lastvals = obj.paramvals.copy()
            lastresiduals = obj.residuals.copy()
            step = self.calc_step(self.damp_JTJ(self.damp))
            err = obj.update(lastvals + step)
            if err < lasterror:
                self.lasterror = lasterror
                self.lastvals[:] = lastvals.copy()
                if self.dobroyden:
                    self.broyden_update_J(obj.paramvals - lastvals,
                                          obj.residuals - lastresiduals)
            else:
                # Put params back, quit:
                obj.update(lastvals)
                return 'stuck'
            CLOG.debug('Run w/ J step: \t{:.3f}'.format(obj.error))
        return 'unconverged'

    def eig_update_J(self, numdir=1):
        """Update J along the `numdir` stiffest eigendirections"""
        CLOG.debug('Eigendirection update.')
        obj = self.optobj
        vl, vc = np.linalg.eigh(obj.JTJ)
        obj.low_rank_J_update(vc[:, -numdir:].T)

    def badstep_update_J(self, badstep, bad_dr):
        """After a bad step, update J along the 2 directions we know are bad.

        Parameters
        ----------
        badstep : n-element numpy.ndarray
            The attempted step direction that failed, same dimension as the
            optobj's paramvals vector.
        bad_dr : d-element numpy.ndarray
            The change in the resiudals after the attempted step, same
            dimension as the optobj's residuals vector. Used to find the
            apparent step direction.
        """
        CLOG.debug('Bad step update.')
        apparent = np.dot(bad_dr, self.optobj.J)  # apparent step
        self.optobj.low_rank_J_update([stp / np.sqrt(np.dot(stp, stp)) for
                                       stp in [badstep, apparent]])


class GroupedOptimizer(object):
    def __init__(self, optimizer_generator):
        """Optimization over groups of optimizers.

        Parameters
        ----------
        optimizer_generator : generator or iterator
            A generator or iterator over optimizer objects to optimize.
            Must return objects with an optimize() method.

        Methods
        -------
        optimize : optimizes each individual optimizer.
        """
        self.optimizer_generator = optimizer_generator

    def optimize(self):
        for lmopt in self.optimizer_generator:
            lmopt.optimize()

