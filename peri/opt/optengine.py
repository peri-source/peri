#FIXME when finish this, update s.gradmodel_e to s.gradmodel_error
# Another way to do this is simply to copy all the code and start deleting things...
# ...wait til you have an organizational structure
import numpy as np
import time
from peri.logger import log
CLOG = log.getChild('optengine')

def _low_mem_mtm(m, step='calc'):
    """
    np.dot(m.T, m) with low mem usage for m non-C-ordered, by using small steps

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
    # -- can make this even faster with pre-allocating arrays, but not worth it
    # right now
    # mt_tmp = np.zeros([step, m.shape[0]])
    # for a in range(0, m.shape[1], step):
        # mx = min(a+step, m.shape[1])
        # mt_tmp[:mx-a,:] = m.T[a:mx]
        # # np.dot(m_tmp, m.T, out=mmt[a:mx])
        # # np.dot(m, m[a:mx].T, out=mmt[:, a:mx])
        # np.dot(m[:,a:mx], mt_tmp[:mx], out=mmt)
    mtm = np.zeros([m.shape[1], m.shape[1]])  #6us
    # m_tmp = np.zeros([step, m.shape[1]])
    for a in range(0, m.shape[1], step):
        mx = min(a+step, m.shape[0])
        # m_tmp[:] = m[a:mx]
        # np.dot(m_tmp, m.T, out=mmt[a:mx])
        # mmt[:, a:mx] = np.dot(m, m[a:mx].T)
        mtm[a:mx,:] = np.dot(m[:,a:mx].T, m)
    return mtm

# The Engine should be flexible enough where it doesn't care whether residuals
# is data-model or model-data, since the OptObj just provides residuals.


# FIXME the OptObj needs to know how to calculate the gradient of the error
# For a state this means (1) the exact gradient of the error after
# updating and (2) a calculated gradient JTr when the the exact grad is wrong
# Also, don't initialize a J array until it is asked for (in the init set
# J=None, then in the update_J if it's not set allocate)

# Things needed for the OptObj:
# 1. raise error if J has nans
# 2. Calculate the gradient of the cost and the residuals, with low memory
#       a. Means that you need a _graderr to be exact for the OptState
# 3. Does the optobj know about JTJ too? certainly not a damped JTJ though.
# 4. If it has a J, it needs to know how to do partial updates of J.
#    - for instance, it needs to know how to do an eigen update of J
#    and a Broyden update of J. And then it needs to know how to intelligently
#    update them (i.e. only Broyden update when we take a good step, not when
#    we try out to infinity on accident)
# 5. Which means it needs to know how to do a rank-1 update
# 6. Consistent notation for J -- J[i,j] = ith residual, j param


class OptObj(object):
    def __init__(self, *args, **kwargs):
        """
        Superclass for optimizing an object, especially for minimizing the
        sum of the squares of a residuals vector.
        This object knows everything that might be needed about the fit
        landscape, including:
        * The residuals vector
        * The current cost (`error`), not necessarily the sum of the squares
          of the residuals vector (e.g. decimated residuals but accurate cost
          or priors) FIXME rename error -> cost?
        * The gradient of the residuals vector, and a way to update it when
          desired
        * The gradient of the cost, and a way to update it with function
          updates / when desired.
        * The current parameter values
        * How to update the object to a new set of parameter values
        It knows nothing about methods to compute steps for those objects,
        however.
        """
        pass

    def update_J(self):
        """
        Updates the Jacobian / gradient of the residuals = gradient of model
        Ideally stored as a C-ordered numpy.ndarray"""
        return None

    def update(self, values):
        """Updates the function to `values`"""
        pass

    def gradcost(self):
        """Returns the gradient of the cost"""
        return 2*np.dot(self.residuals, self.J)

    @property
    def error(self):
        r = self.residuals.copy()
        return np.dot(r,r)

    @property
    def residuals(self):
        """Returns the residuals = model-data"""
        pass

    @property
    def paramvals(self):
        """Returns the current value of the parameters"""
        pass

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
            if not np.isclose(np.dot(d,d), 1.0, atol=1e-10):
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
                self.J[:,b] += d[b] * delta_vals
        if np.isnan(self.J.sum()):
            raise RuntimeError('J has nans')
        self._calcjtj()

    def _calcjtj(self):
        self.JTJ = _low_mem_mtm(self.J)
        if np.isnan(self.JTJ.sum()):
            raise RuntimeError('JTJ has nans')

    def find_expected_error(self, delta_params='perfect'):
        """
        Returns the error expected after an update if the model were linear.

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
        # If the model were linear, then the cost would be quadratic,
        # with Hessian 2*`self.JTJ` and gradient `grad`
        expected_error = (self.error + np.dot(grad, delta_params) +
                np.dot(np.dot(self.JTJ, delta_params), delta_params))
        return expected_error

    def calc_model_cosine(self):
        """
        Calculates the cosine of the residuals with the model, based on the
        expected error of the model.

        Returns
        -------
        abs_cos : numpy.float64
            The absolute value of the model cosine.

        Notes
        -----
        The model cosine is defined in terms of the geometric view of
        curve-fitting, as a model manifold embedded in a high-dimensional
        space. The model cosine is the cosine of the residuals vector
        with its projection on the tangent space: :math:`cos(phi) = |P^T r|/|r|`
        where :math:`P^T` is the projection operator onto the model manifold
        and :math:`r` the residuals.

        Rather than doing the SVD, we get this from the expected error of
        the model if it were linear. We use that expected error and the
        current error to calculate a model sine, and use that to get a model
        cosine
        """
        expected_error = self.find_expected_error(delta_params='perfect')
        model_sine_2 = expected_error / self.error  #error = distance^2
        abs_cos = np.sqrt(1 - model_sine_2)
        return abs_cos

#List:
# 1. Check
# 2. Check (no decimation)
# 3. Check
# 4. Not to be implemented; for optimizer.
# 5. Check
# 6. Check
class OptFunction(OptObj):
    def __init__(self, func, data, paramvals, dl=1e-7):
        """
        OptObj for a function

        Parameters
        ----------
            func : callable
                Must return a np.ndarray-like with syntax func(data, *params)
                and return an object of the same size as ``data``
            data : flattened numpy.ndarray
                Passed to the func. Must return a valid ``np.size``
            paramvals : numpy.ndarray
                extra args to pass to the function.
            dl : Float, optional
                Step size for finite-difference (2pt stencil) derivative
                approximation. Default is 1e-7
        """
        self.func = func
        self.data = data
        self._model = np.zeros_like(data)
        self._paramvals = np.array(paramvals).reshape(-1)
        self.J = None # we don't create J until we need to
        self.dl = dl

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
            self.update(p0+dp)
            r1 = self.residuals.copy()
            self.J[:, i] = (r1-r0) / self.dl
        if np.isnan(self.J.sum()):
            raise RuntimeError('J has nans')
        self._calcjtj()
        #And we put params back:
        self.update(p0)

    def update(self, values):
        self._paramvals[:] = values
        self._model = self.func(self._paramvals)
        return self.error

    @property
    def residuals(self):
        return np.ravel(self._model - self.data)

    @property
    def paramvals(self):
        return self._paramvals.copy()

def get_residuals_update_tile(st, params, vals=None):
    """
    Finds the update tile of a state in the residuals (unpadded) image.

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
    if vals is None:
        vals = st.get_values(params)
    itile = st.get_update_io_tiles(params, vals)[1]
    inner_tile = st.ishape.intersection([st.ishape, itile])
    return inner_tile.translate(-st.pad)

def decimate_state(state, nparams, decimate=1, min_redundant=20, max_mem=1e9):
    """
    Given a state, returns set of indices """
    pass # FIXME

# Only works for image state because the residuals update tile only works
# for image states -- get_update_io_tile is only for ImageState
class OptImageState(OptObj):
    def __init__(self, state, params, dl=3e-6, inds=None):
        """
        OptObj for a peri.states.ImageState

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
	"""
        self.state = state
        self.params = params
        self.dl = dl
        self.tile = get_residuals_update_tile(state, params)
        if inds is None:
	    self.inds = slice(None)
	else:
	    self.inds = inds
        self.J = None

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
        self.J[:] = np.transpose(self.state.gradmodel(params=self.params,
                rts=True, dl=self.dl, slicer=self.tile.slicer, inds=self.inds))
        CLOG.debug('Calcualted J:\t{} s'.format(time.time() - start))
        self._calcjtj()

    @property
    def residuals(self):
        # FIXME residuals = model - data
        return -self.state.residuals[self.tile.slicer].ravel()[self.inds]

    @property
    def error(self):
        return self.state.error

    @property
    def paramvals(self):
        return np.copy(self.state.get_values(self.params))

    def update(self, values):
        self.state.update(self.params, values)
        return self.error


# Things needed for the engine:
# 1. 1 class, only relies on OptObj
# 2. Clear termination criterion
#   a. Terminate when stuck.
#   b. Returned termination flags (e.g. completed, stuck, maxiter)
# 3. User-supplied damping that scales with problem size
#   a. Levenberg and marquardt modes
#   b. vectorial modes
#   c. scaled modes
# 5. Only 1 run mode.

# Old things that need to be kept for the engine
# 1. Low-memory-overhead dots for JTJ and J*residuals
# 2. acceleration options -- done
# 3. Broyden option, default as true
# 4. Internal run -- partially done but not implemented

# Things that maybe should be a generic module / exterior callable / passed
# model-like object.
# 1. vector damping                             |
# 2. increase/decrease damping factors....      |   DAMPING
# 3. damping modes....                          |
# 4. acceleration options?
# 5. Update-J options (e.g. eig or guessing an update after failure

# Things that should be generic:
# 1. Should not matter whether or not residuals = model-data or data-model

# Possible new features to add
# 1. Guess updates when stuck / failed step?
# 2. A better way to update the damping significantly when needed.


# Possible features to remove:
# 1. Subblock runs

#Needs:
class LMOptimizer(object):
    def __init__(self, optobj, damp=1.0, dampdown=8., dampup=3., nsteps=(1,2),
            accel=True, dobroyden=True, exptol=1e-7, costol=1e-5, errtol=1e-7,
            fractol=1e-7, paramtol=1e-7, maxiter=2):
        """
        Parameters
        ----------
        optobj : OptObj instance
            The OptObj to optimize.
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
        self.dampmode = lambda JTJ, damp: JTJ + np.eye(JTJ.shape[0]) * damp
        self.accel = accel
        self.dobroyden = dobroyden

        # Termination dict:
        self.maxiter = maxiter
        self.term_dict = {'errtol':errtol, 'fractol':fractol,
                'paramtol':paramtol, 'exptol':exptol, 'costol':costol}

    def optimize(self):
        """Runs the optimization"""
        for _ in range(self.maxiter):
            CLOG.debug('Start loop {}: \t{}'.format(_, self.optobj.error))
            # Most generic algorithm is:
            # 1. Update J, JTJ
            self.optobj.update_J()
            # 2. Calculate & take a step -- distinction might be blurred
            flag = self.take_initial_step()
            if flag == 'stuck':
                return flag
            # 3. If desired, take more steps
            flag = self.take_additional_steps()
            #       - could be run with J
            #       - could be run with J with quick updates when stuck
            #           (quick = eig, directional)
            # 4. Repeat til termination
            if np.any(list(self.check_completion().values())):
                return 'completed'
        else: # for-break-else, with the terminate
            # something about didn't converge maybe
            return 'unconverged'

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
        CLOG.debug('{}'.format([lasterror]+ errs))
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
                    CLOG.debug('Increased damping {}x, {}'.format(_, err))
                    break
            else:  # for-break-else, failed to increase damping
                # update function to previous value, terminate?
                obj.update(lastvals)
                CLOG.warn('Stuck!')
                return 'stuck'
        # If we're still here, we've taken a good step, so broyden update:
        CLOG.debug('Initial step: \t{}'.format(obj.error))
        self.lasterror = lasterror
        self.lastvals[:] = lastvals.copy()
        if self.dobroyden:
            self.broyden_update_J(  obj.paramvals - lastvals,
                                    obj.residuals - lastresiduals)
        return 'run'

    def take_additional_steps(self):
        """Takes additional steps"""
        # Right now just run with J, but could be a more complicated procedure.
        return self.run_with_J()

    def damp_JTJ(self, damp):
        # One possible different option is to pass a self.dampmode as a function
        # which takes JTJ, damping and returns a damped JTJ
        # -- right now defined explicitly in the init
        return self.dampmode(self.optobj.JTJ, damp)

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
            CLOG.debug('Run w/ J step: \t{}'.format(obj.error))
        return 'unconverged'

    # def another_run(self, numdir=1):
        # flag = run_with_J()
        # if flag == 'unconverged':
            # self.eig_update_J(numdir=numdir)
            # self.badstep_updat_J(badstep, badresiduals)
            # -- need to get the badresiduals w/o a function update though

    def eig_update_J(self, numdir=1):
        """Update J along the `numdir` stiffest eigendirections"""
        CLOG.debug('Eigendirection update.')
        obj = self.optobj
        vl, vc = np.linalg.eigh(obj.JTJ)
        obj.low_rank_J_update(vc[:, -numdir:].T)

    def broyden_update_J(self, direction, dr):
        """Update J along `direction` for change in residuals `dr` (both 1D)"""
        CLOG.debug('Broyden update.')
        nrm = np.sqrt(np.dot(direction, direction))
        d0 = direction / nrm
        vals = dr / nrm
        self.optobj.low_rank_J_update(d0.reshape(1,-1), vals.reshape(1,-1))

    def badstep_update_J(self, badstep, bad_dr):
        """
        After a bad step, update J along the 2 directions we know are bad.

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
        self.optobj.low_rank_J_update([stp / np.sqrt(np.dot(stp,stp)) for stp
                in [badstep, apparent]])

    def calc_step(self, dampedJTJ):
        grad = self.optobj.gradcost()
        # could be augmented to include acceleration etc:
        # corr = self.calc_accel_correction(dampedJTJ, initialstep)
        simple = self.calc_simple_LM_step(dampedJTJ, grad)
        if self.accel:
            return simple + self.calc_accel_correction(dampedJTJ, simple)
        else:
            return simple

    def calc_simple_LM_step(self, dampedJTJ, grad):
        return np.linalg.lstsq(dampedJTJ, -0.5*grad, rcond=self._rcond)[0]

    def check_completion(self):
        """Return a bool of whether or not optimization has converged"""
        d = self.get_convergence_stats()
        keys = [
                ['derr',     'errtol'],
                ['fracerr',  'fractol'],
                ['dvals',    'paramtol'],
                ['modelcos', 'costol'],
                ['exp_derr', 'exptol'],
                ]
        return {k2:d[k1] < self.term_dict[k2] for k1, k2 in keys}

    def get_convergence_stats(self):
        """Returns a dict of termination info"""
        obj = self.optobj
        d = {
            'derr' : self.lasterror - obj.error,
            'fracerr' : (self.lasterror - obj.error),
            'dvals' : np.abs(self.lastvals - obj.paramvals).max(),
            'modelcos':obj.calc_model_cosine(),
            'exp_derr':obj.error - obj.find_expected_error(),
            }
        return d

    def calc_accel_correction(self, dampedJTJ, initialstep):
        """
        Geodesic acceleration correction to the LM step.

        Parameters
        ----------
            dampedJTJ : numpy.ndarray
                The damped JTJ used to calculate the initial step.
            initialstep : numpy.ndarray
                The initial LM step.

        Returns
        -------
            corr : numpy.ndarray
                The correction to the original LM step.
        """
        #Get the derivative:
        obj = self.optobj
        p0 = obj.paramvals.copy()  # FIXME this is probably wrong
        _ = obj.update(p0)
        rm0 = obj.residuals.copy()
        _ = obj.update(p0 + initialstep)
        rm1 = obj.residuals.copy()
        _ = obj.update(p0 - initialstep)
        rm2 = obj.residuals.copy()
        der2 = (rm2 + rm1 - 2*rm0)

        correction = np.linalg.lstsq(dampedJTJ, np.dot(der2, obj.J),
                                    rcond=self._rcond)[0]
        correction *= -0.5
        return correction



if __name__ == '__main__':
    # Test an OptObj:
    log.set_level('debug')
    from peri.opt import opttest
    f = opttest.increase_model_dimension(opttest.rosenbrock)
    o = OptFunction(f, f(np.array([1.0, 1.0])), np.zeros(2), dl=1e-7)
    l = LMOptimizer(o, maxiter=1)
    l.optimize()

    # also works for image state, w/o decimation...

    # so: Make it work for decimation, make opt procedures.
    # TODO: 
    # Decimation for optstate -- should be just inds=inds and calculating them
    # GroupedOptimizer
    # - Decimate by paramters
    # - Line / quadratic / etc minimization
    # - Probably works via a generator which gets called each time in a loop?

