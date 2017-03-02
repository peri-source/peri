#FIXME when finish this, update s.gradmodel_e to s.gradmodel_error
# Another way to do this is simply to copy all the code and start deleting things...
# ...wait til you have an organizational structure
import numpy as np

# The Engine should be flexible enough where it doesn't care whether residuals
# is data-model or model-data, since the OptObj just provides residuals.


# FIXME the OptObj needs to know how to calculate the gradient of the error
# For a state this means (1) the exact gradient of the error after
# updating and (2) a calculated gradient JTr when the the exact grad is wrong
# Also, don't initialize a J array until it is asked for (in the init set
# J=None, then in the update_J if it's not set allocate)

# Things needed for the OptObj:
# 1. raise error if J has nans
        # if np.any(np.isnan(self.JTJ)):
            # raise FloatingPointError('J, JTJ have nans.')
# 2. Calculate the gradient of the cost and the residuals, with low memory
#       a. Means that you need a _graderr to be exact for the OptState
# 3. Does the optobj know about JTJ too? certainly not a damped JTJ though.
# 4. If it has a J, it needs to know how to do partial updates of J.
#    - for instance, it needs to know how to do an eigen update of J
#    and a Broyden update of J. And then it needs to know how to intelligently
#    update them (i.e. only Broyden update when we take a good step, not when
#    we try out to infinity on accident)
# 5. Which means it needs to know how to do a rank-1 update
#
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
          updates.
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
        pass

    @property
    def error(self):
        """Returns the error"""
        pass

    @property
    def J(self):
        """returns J. Don't make it a copy"""
        pass

    @property
    def residuals(self):
        """Returns the residuals = model-data"""
        pass

    @property
    def paramvals(self):
        """Returns the current value of the parameters"""
        pass

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
        self.J = np.zeros([data.size, self.params.size], dtype='float')
        self.dl = dl

    def update_J(self):
        r0 = self.residuals.copy()
        p0 = self._paramvals
        dp = np.zeros_like(p0)
        for i in range(p0.size):
            dp *= 0
            dp[i] = dl
            self.update_function(p0+dp)
            r1 = self.residuals.copy()
            self.J[:, i] = (r1-r0)/dl

    def gradcost(self):
        return np.dot(self.J.T, self.residuals)  # should be a lowmem dot but w/e

    @property
    def error(self):
        r = self.residuals.copy()
        return np.dot(r,r)

    def update(self, values):
        self.values[:]
        self._model = self.func(self.data, *self.values)

    @property
    def residuals(self):
        return np.ravel(self._model - self.data)

    @property
    def paramvals(self):
        return self._paramvals.copy()

# Things needed for the engine:
# 1. 1 class, only relies on OptObj
# 2. Clear termination criterion
#   a. Terminate when stuck.
#   b. Returned termination flags (e.g. completed, stuck, maxiter)
# 3. User-supplied damping that scales with problem size
# 4. Consistent notation for J -- J[i,j] = ith residual, j param
# 5. Only 1 run mode.

# Old things that need to be kept for the engine
# 1. Low-memory-overhead dots for JTJ and J*residuals
# 2. acceleration options?
# 3. Broyden option, default as true
# 4. Internal run.

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

def _low_mem_mtm(m, step='calc'):
    """np.dot(m.T, m) with low mem usage, by doing it in small steps
    Default step size is 1% additional mem overhead."""
    if not m.flags.c_contiguous:
        raise ValueError('m must be C ordered for this to work with less mem.')
    if step == 'calc':
        step = np.ceil(1e-2 * self.J.shape[0]).astype('int')
    # -- can make this even faster with pre-allocating arrays, but not worth it
    # right now
    # mt_tmp = np.zeros([step, m.shape[0]])
    # for a in range(0, m.shape[1], step):
        # mx = min(a+step, m.shape[1])
        # mt_tmp[:mx-a,:] = m.T[a:mx]
        # # np.dot(m_tmp, m.T, out=mmt[a:mx])
        # # np.dot(m, m[a:mx].T, out=mmt[:, a:mx])
        # np.dot(m[:,a:mx], mt_tmp[:mx], out=mmt)
    mmt = np.zeros([m.shape[1], m.shape[1]])  #6us
    # m_tmp = np.zeros([step, m.shape[1]])
    for a in range(0, m.shape[1], step):
        mx = min(a+step, m.shape[0])
        # m_tmp[:] = m[a:mx]
        # np.dot(m_tmp, m.T, out=mmt[a:mx])
        # mmt[:, a:mx] = np.dot(m, m[a:mx].T)
        mmt[a:mx,:] = np.dot(m[:,a:mx].T, m)
    return mmt

def _low_mem_mmt(m, step='calc'):
    """np.dot(m, m.T) with low mem usage, by doing it in small steps
    Default step size is 1% more mem overhead"""
    if not m.flags.c_contiguous:
        raise ValueError('m must be C ordered for this to work with less mem.')
    if step == 'calc':
        step = np.ceil(1e-2 * self.J.shape[1]).astype('int')
    mmt = np.zeros([m.shape[0], m.shape[0]])  #6us
    for a in range(0, m.shape[0], step):
        mx = min(a+step, m.shape[1])
        mmt[:, a:mx] = np.dot(m, m[a:mx].T)
    return mmt



class LMOptimizer(object):
    def __init__(self, optobj):
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
        self._rcond = 1e-13  # rcond for leastsq step, after damping
        pass

    def optimize(self):
        """Runs the optimization"""
        while not self.check_terminate():

            # Most generic algorithm is:
            # 1. Update J, JTJ
            self.optobj.update_J()....???
            # 2. Calculate & take a step -- distinction might be blurred
            #       a. Calculate N steps
            #               i.  The damping will be updated during this process
            #               ii. Might involve acceleration etc steps
            #       b. Pick the best step that is also good
            #               i.  If good, take that step
            #               ii. If not good, increase damping somehow til step
            #                   is good
            # 3. If desired, take more steps
            #       - could be run with J
            #       - could be run with J with quick updates when stuck
            #           (quick = eig, directional)
            # 4. Repeat til termination
            raise NotImplementedError

    # def update_J():
        # # 1. Actually update J:
        # self.optobj.update_J()
        # # 2. Calculate JTJ with low mem overhead:
        # self.JTJ = _low_mem_mtm(self.optobj.J)
        # # self._exp_err = self.error - self.find_expected_error(delta_params='perfect')
        # pass  # expected error should be useful.... might need to be done here? Or only when desired?

    def calc_simple_LM_step(self, dampedJTJ, grad):
        return np.linalg.leastsq(damped_JTJ, -0.5*grad, rcond=self._rcond)[0]

    def check_terminate():
        """Return a bool of whether or not something terminated"""
        raise NotImplementedError

    def calc_accel_correction(self, damped_JTJ, initialstep):
        """
        Geodesic acceleration correction to the LM step.

        Parameters
        ----------
            damped_JTJ : numpy.ndarray
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
        p0 = obj.paramvals  # FIXME this is probably wrong
        _ = obj.update(p0)
        rm0 = obj.residuals.copy()
        _ = obj.update(p0)
        rm1 = obj.residuals.copy()
        _ = obj.update(p0)
        rm2 = obj.residuals.copy()
        der2 = (rm2 + rm1 - 2*rm0)

        correction = np.linalg.lstsq(damped_JTJ, np.dot(self.J, der2),
                                    rcond=self.min_eigval)[0]
        correction *= -0.5
        return correction




















