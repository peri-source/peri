"""
A series of functions for testing curve-fitting behavior. Most of these
functions are taken from wikipedia:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
import numpy as np


def himmelblau(xy):
    """
    Himmelblau's function, as a set of residuals (cost = sum(residuals**2))

    The standard Himmelbau's function is with data as [11, 7], and four
    minimum at (3.0, 2.0), ~(-2.8, 3.1), ~(-3.8, -3.3), ~(3.6, -1.8).
    Himmelblau's function is a quadratic model in both x and y. Its data-
    space dimension (2) is equal to its model-space dimension (2), so
    there is only parameter-effect curvature.

    Parameters
    ----------
        - xy : 2-element list-like
            The x,y parameters of the model.

    Returns
    -------
        2-element list-like
            The residuals of the model.
    Notes
    ------
    https://en.wikipedia.org/wiki/Himmelblau%27s_function
    """
    x, y = xy
    r1 = x*x + y
    r2 = y*y + x
    return np.array([r1, r2])


def simple_sphere(xy):
    """
    A simple sphere, as a set of residuals (cost = sum(residuals**2))

    Simply returns the input parameters. A linear, uncoupled model.

    Parameters
    ----------
        - xy : 2-element list-like
            The x,y parameters of the model.

    Returns
    -------
        - xy : 2-element list-like
            The residuals, which are just xy
    """
    return xy


def rosenbrock(xy, A=10):
    """
    The original rosenbrock banana function, as a set of residuals
    (cost = sum(residuals**2))

    The original function is with data = [1,0] and A=10, with a global
    minimum at (1,1)=(data[0], data[0]^2). It is a coupled model,
    quadratic in x and linear in y. Its data-space dimension (2) is equal
    to its model-space dimension (2), so there is only parameter-effect
    curvature.

    Parameters
    ----------
        - xy : 2-element list-like
            The x,y parameters of the model.

    Returns
    -------
        2-element list-like
            The residuals of the model.
    """
    x, y = xy
    r1 = x
    r2 = A*(y-x*x)
    return np.array([r1, r2])


def rosenbrock_gen(xy, A=10, order=3):
    """
    A generalized rosenbrock banana function, as a set of residuals
    (cost = sum(residuals**2))

    The Rosenbrock function, generalized from a quadratic model to a
    higher-order nonlinearity. The residuals are
        r1 = xy[0]
        r2 = xy[1] - xy[0]^n/n
    The original function is with data = [1,0] and order=2, and has a
    single minimum at (x,y) = (1, 1/order). The model is coupled,
    polynomial in x and linear in y. Its data-space dimension (2) is
    equal to its model-space dimension (2), so there is only parameter-
    effect curvature. (See M Transtrum et al, PRE 2011)

    Parameters
    ----------
        - xy : 2-element list-like
            The x,y parameters of the model.
        - order : Int, optional
            The order of the model nonlinearity. Default is 3

    Returns
    -------
        2-element list-like
            The residuals of the model.
    """
    x, y = xy
    r1 = x
    r2 = A*(y-x**order / order)
    return np.array([r1, r2])


def rosenbrock_dd(xd, A=10):
    """
    A higher-dimensional modification of the rosenbrock function, as a
    set of residuals (cost = sum(residuals**2))

    The standard modified function is with data = zeros(2d-2), with a
    global minimum at xd = ones(d). It is a coupled model, quadratic in
    the first d-1 parameters. Its data-space dimension (2d-2) is greater
    than its model-space dimension (d) for d > 2.

    Parameters
    ----------
        - xd : d-element list-like
            The x,y parameters of the model.

    Returns
    -------
        (2d-2)-element list-like
            The residuals of the model.

    Notes
    -----
        Based on the multidimensional variation
            f(x) = \sum_{i=1}^{N-1} 100(x_{i+1} - x_i&2) + (1-x_i)^2
    This gives a data space of dimension 2*(N-1) =2N-2 and a parameter
    space of dimension N. According to wikipedia, has 1 minimia for N=3
    at (1,1,1), 2 minima for 4<=N<=7. See:
    https://en.wikipedia.org/wiki/Rosenbrock_function#Multidimensional_generalisations
    """
    xp = xd[1:]
    xi = xd[:-1]
    r1 = A*(xp - xi*xi)
    r2 = 1 - xi
    return np.append(r1, r2)


def rosenbrock_gendd(xd, A=10, order=3):
    """
    A higher-dimensional modification of a generalized rosenbrock
    function, as a set of residuals (cost = sum(residuals**2))

    The standard modified function is with data = zeros(N), with a
    global minimum at xd=ones(d). This function is a coupled model,
    polynomial in the first d-1 parameters. Its data-space dimension
    (2d-2) is greater than its model-space dimension (d) for d > 2.

    Parameters
    ----------
        - xd : d-element list-like
            The x,y parameters of the model.
        - order : Int, optional
            The order of the model nonlinearity. Default is 3

    Returns
    -------
        (2d-2)-element list-like
            The residuals of the model.

    Notes
    -----
        Based on the multidimensional variation
            f(x) = \sum_{i=1}^{N-1} 100(x_{i+1} - x_i&2) + (1-x_i)^2
    This gives a data space of dimension 2*(N-1) =2N-2 and a parameter
    space of dimension N. According to wikipedia, the quadratic version
    has 1 minimia for N=3 at (1,1,1), 2 minima for 4<=N<=7. See:
    https://en.wikipedia.org/wiki/Rosenbrock_function#Multidimensional_generalisations
    """
    xp = xd[1:]
    xi = xd[:-1]
    r1 = A*(xp - xi**order)
    r2 = 1 - xi
    return np.append(r1, r2)


def beale(xy):
    """
    The Beale function, as a set of residuals (cost = sum(residuals**2))

    The standard Beale's function is with data as [1.5, 2.25, 2.625],
    and has a global minima at (3, 0.5). Beale's function is a coupled
    model, linear in x and quartic in y. Its data-space dimension (3) is
    greater than its model-space dimension (2).

    Parameters
    ----------
        - xy : 2-element list-like
            The x,y parameters of the model.

    Returns
    -------
        3-element list-like
            The residuals of the model.
    """
    x, y = xy
    r1 = x - x*y
    r2 = x - x*y*y
    r3 = x - x*y*y*y
    return np.array([r1, r2, r3])


def booth(xy):
    """
    The Booth's function, as a set of residuals (cost = sum(residuals**2))

    The standard Booth function is with data as [7, 5], and has a single
    global minimum at (1,3). It is a coupled linear model, with the
    parameter and data space both 2-dimensional.

    Parameters
    ----------
        - xy : 2-element list-like
            The x,y parameters of the model.

    Returns
    -------
        2-element list-like
            The residuals of the model.
    """
    x, y = xy
    r1 = x + 2*y
    r2 = y + 2*x
    return np.array([r1, r2])

# def increase_model_dimension(func):
    # """
    # Something that takes a [d,N] dimensional model and transforms to a
    # [xD, xN] where x is an int..? coupling would be nice...
    # """
    # pass
