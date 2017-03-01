import numpy as np

# The Engine should be flexible enough where it doesn't care whether residuals
# is data-model or model-data, since the OptObj just provides residuals.

class OptObj(object):
    def __init__(self, *args, **kwargs):
        """Empty class for structure"""
        pass

    def update_J(self):
        """
        Updates the Jacobian / gradient of the residuals = gradient of model
        Ideally stored as a C-ordered numpy.ndarray"""
        return None

    def update(self, values):
        """Updates the function to `values`"""
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
# 5. Gradient of cost vs gradient of error -- _graderr directly from state

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



class LMEngine(object):
    def __init__(self, optobj):
        """
        Parameters
        ----------
        optobj : OptObj instance
            The OptObj to optimize.
        """
        self.optobj = optobj
        pass

    def optimize(self):
        """Runs the optimization"""
        while not self.check_terminate():

            # Most generic algorithm is:
            # 1. Update J, JTJ
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

    def update_J():
        # 1. Actually update J:
        self.optobj.update_J()
        # 2. Calculate JTJ with low mem overhead:
        self.JTJ = _low_mem_mtm(self.optobj.J)
        if np.any(np.isnan(self.JTJ)):
            raise FloatingPointError('J, JTJ have nans.')
        # 3. Update some flags:
        # self._exp_err = self.error - self.find_expected_error(delta_params='perfect')
        self._freshjtj = True  # basically only needed for the whole self._graderr vs calc_grad() bit....
        # -- so delete it! Include graderr in optobj?

    def check_terminate():
        """Return a bool of whether or not something terminated"""
        raise NotImplementedError





















