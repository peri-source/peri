from builtins import range, object

import numpy as np


class BarnesInterpolation1D(object):
    def __init__(self, x, d, filter_size=None, iterations=4, clip=False,
                 clipsize=3, damp=0.95, blocksize=None, donorm=True):
        """1-d barnes interpolation. Give data points d at locations x.

        See [1]_, equations 1-7 for implementation.

        Parameters
        ----------
        x : ndarray, 1-dimensional
            input positions, x values for data points

        d : ndarray, 1-dimensional
            input values, y values for data points

        filter_size : float, optional.
            control parameter for weight function (sigma), should be the
            average data spacing. Defaults to the average distance for
            sorted ``x``.

        iterations : integer, optional
            how many iterations to perform. only two needed with a high damping
            Defaults to 4

        clip : boolean, optional
            whether to clip the number of data points used by the filtersize
            Default is False

        clipsize : float, optional
            Total clipsize is determined by clipsize * filter_size
            Default is 3

        damp : float, optional
            the damping parameter used in accelerating convergence. Default
            is 0.95

        blocksize : Int or None, optional
            Memory-based performance feature. For very large ``x`` or ``d``
            calculating the distance matrix between each point can require
            excessive memory overhead. To avoid this, set ``blocksize`` to
            a moderate int, causing the Barnes to compute the distance
            matrix in blocks. Does not change the final results. Default is
            None, which uses all the data all at once, using lots of mem.

        donorm : bool, optional
            If False, uses an old, incorrect method to evaluate the interpolant
            rather than the correct version. Default is True. If you're using
            this, set it to True.

        Examples
        --------
        >>> import numpy as np
        >>> x_known = np.linspace(0, np.pi, 10)
        >>> d_known = np.sin(x_known)
        >>> barnes = BarnesInterpolation1D(x_known, d_known)
        >>> x_check = np.linspace(0, np.pi, 11)
        >>> np.allclose(barnes(x_check), np.sin(x_check), atol=1e-2)
        True

        References
        ----------
        .. [1] S. E. Koch, M. DesJardins, P. J. Kocin, J. Climate Appl.
                Meteor. 22 1487-1503 (1983)
        """
        self.x = x
        self.d = d
        self.damp = damp
        self.clip = clip
        self.iterations = iterations
        self.donorm = donorm
        self.blocksize = blocksize

        if filter_size is None:
            self.filter_size = self._default_filter_size()
        else:
            self.filter_size = filter_size

        self.clipsize = clipsize * self.filter_size

    def _default_filter_size(self):
        return (self.x[1:] - self.x[:-1]).mean()/2

    def _distance_matrix(self, a, b):
        """Pairwise distance between each point in `a` and each point in `b`"""
        return (a[:, None] - b[None, :])**2

    def _weight(self, rsq, sigma=None):
        """weighting function for Barnes"""
        sigma = sigma or self.filter_size

        if not self.clip:
            o = np.exp(-rsq / (2*sigma**2))
        else:
            o = np.zeros(rsq.shape, dtype='float')
            m = (rsq < self.clipsize**2)
            o[m] = np.exp(-rsq[m] / (2*sigma**2))
        return o

    def __call__(self, rvecs):
        """
        Get the values interpolated at positions rvecs
        """
        if self.donorm:
            return self._newcall(rvecs)
        else:
            return self._oldcall(rvecs)

    def _eval_firstorder(self, rvecs, data, sigma):
        """The first-order Barnes approximation"""
        if not self.blocksize:
            dist_between_points = self._distance_matrix(rvecs, self.x)
            gaussian_weights = self._weight(dist_between_points, sigma=sigma)
            return gaussian_weights.dot(data) / gaussian_weights.sum(axis=1)
        else:
            # Now rather than calculating the distance matrix all at once,
            # we do it in chunks over rvecs
            ans = np.zeros(rvecs.shape[0], dtype='float')
            bs = self.blocksize
            for a in range(0, rvecs.shape[0], bs):
                dist = self._distance_matrix(rvecs[a:a+bs], self.x)
                weights = self._weight(dist, sigma=sigma)
                ans[a:a+bs] += weights.dot(data) / weights.sum(axis=1)
            return ans

    def _newcall(self, rvecs):
        """Correct, normalized version of Barnes"""
        # 1. Initial guess for output:
        sigma = 1*self.filter_size
        out = self._eval_firstorder(rvecs, self.d, sigma)
        # 2. There are differences between 0th order at the points and
        #    the passed data, so we iterate to remove:
        ondata = self._eval_firstorder(self.x, self.d, sigma)
        for i in range(self.iterations):
            out += self._eval_firstorder(rvecs, self.d-ondata, sigma)
            ondata += self._eval_firstorder(self.x, self.d-ondata, sigma)
            sigma *= self.damp
        return out

    def _oldcall(self, rvecs):
        """Barnes w/o normalizing the weights"""
        g = self.filter_size

        dist0 = self._distance_matrix(self.x, self.x)
        dist1 = self._distance_matrix(rvecs, self.x)

        tmp = self._weight(dist0, g).dot(self.d)
        out = self._weight(dist1, g).dot(self.d)

        for i in range(self.iterations):
            out = out + self._weight(dist1, g).dot(self.d - tmp)
            tmp = tmp + self._weight(dist0, g).dot(self.d - tmp)
            g *= self.damp
        return out


class BarnesInterpolationND(BarnesInterpolation1D):
    def __init__(self, *args, **kwargs):
        """
        A class for barnes interpolation in N dimensions.

        Parameters
        ----------
        x : ndarray, 2-dimensional
            input positions, x values for data points. x[i] is the ith position

        d : ndarray, 1-dimensional
            input values, y values for data points. Same number of points as
            x has positions.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> x_known = np.random.randn(20, 2)
        >>> func = lambda x: np.exp(-(x[:, 0]**2 + x[:, 1]**2))
        >>> d_known = func(x_known)
        >>> barnes_nd = BarnesInterpolationND(x_known, d_known, iterations=35)
        >>> np.allclose(barnes_nd(x_known), func(x_known), atol=1e-2)
        True

        See Also
        --------
        BarnesInterpolation1D
        """
        super(BarnesInterpolationND, self).__init__(*args, **kwargs)

    def _distance_matrix(self, a, b):
        """Pairwise distance between each point in `a` and each point in `b`"""
        def sq(x): return (x * x)
        # matrix = np.sum(map(lambda a,b: sq(a[:,None] - b[None,:]), a.T,
        #   b.T), axis=0)
        # A faster version than above:
        matrix = sq(a[:, 0][:, None] - b[:, 0][None, :])
        for x, y in zip(a.T[1:], b.T[1:]):
            matrix += sq(x[:, None] - y[None, :])
        return matrix

    def _default_filter_size(self):
        def dist(x): return np.sqrt(np.sum(x*x, axis=1))
        return dist(self.x[1:] - self.x[:-1]).mean()/2


class ChebyshevInterpolation1D(object):
    def __init__(self, func, args=(), window=(0., 1.), degree=3, evalpts=4):
        """A 1D Chebyshev approximation / interpolation for an ND function,
        approximating (N-1)D in in the last dimension.

        Parameters
        ----------
        func : callable
            A function that takes scalar arguments (1D) and returns a N
            dimensional array corresponding to that scalar. Make it such that,
            for an array x, f(x)[.....,a] corresponds to f(x[a])

        args : tuple [optional]
            extra arguments to pass to func

        window : tuple (length 2)
            The bounds of the function over which we desire the interpolation

        degree : integer
            Degree of the Chebyshev interpolating polynomial

        evalpts : integer
            Number of Chebyshev points to evaluate the function at

        Examples
        --------
        >>> import numpy as np
        >>> func = lambda x: np.sin(x)
        >>> cheb = ChebyshevInterpolation1D(func, window=(0, np.pi), degree=9,
        ...                                 evalpts=11)
        >>> np.allclose(cheb(1.0), np.sin(1.0), atol=1e-7)
        True
        """
        self.args = args
        self.func = func
        self.window = window
        self.set_order(evalpts, degree)

    def _x2c(self, x):
        """ Convert windowdow coordinates to cheb coordinates [-1,1] """
        return ((2 * x - self.window[1] - self.window[0]) /
                (self.window[1] - self.window[0]))

    def _c2x(self, c):
        """ Convert cheb coordinates to windowdow coordinates """
        return 0.5 * (self.window[0] + self.window[1] +
                      c * (self.window[1] - self.window[0]))

    def _construct_coefficients(self):
        """Calculate the coefficients based on the func, degree, and
        interpolating points.
        _coeffs is a [order, N,M,....] array

        Notes
        -----
        Moved the -c0 to the coefficients defintion
        app -= 0.5 * self._coeffs[0] -- moving this to the coefficients
        """
        coeffs = [0]*self.degree

        N = float(self.evalpts)

        lvals = np.arange(self.evalpts).astype('float')
        xpts = self._c2x(np.cos(np.pi*(lvals + 0.5)/N))
        fpts = np.rollaxis(self.func(xpts, *self.args), -1)

        for a in range(self.degree):
            inner = [
                fpts[b] * np.cos(np.pi*a*(lvals[b]+0.5)/N)
                for b in range(self.evalpts)
            ]
            coeffs[a] = 2.0/N * np.sum(inner, axis=0)

        coeffs[0] *= 0.5
        self._coeffs = np.array(coeffs)

    def set_order(self, evalpts, degree):
        if evalpts < degree:
            raise ValueError("Number of Chebyshev points must be > degree")

        self.evalpts = evalpts
        self.degree = degree

        self._construct_coefficients()

    @property
    def coefficients(self):
        return self._coeffs.copy()

    def tk(self, k, x):
        """
        Evaluates an individual Chebyshev polynomial `k` in coordinate space
        with proper transformation given the window
        """
        weights = np.diag(np.ones(k+1))[k]
        return np.polynomial.chebyshev.chebval(self._x2c(x), weights)

    def __call__(self, x):
        """
        Approximates `func` at the coordinates x, which must be in the window.

        .. math::
            f(x) = \sum_{k=0}^{N-1} c_k T_k(x) - co/2

        Output is in the format [A,...,x]
        """
        return np.polynomial.chebyshev.chebval(
            self._x2c(x), self._coeffs, tensor=True)
