*************************
Optimization and sampling
*************************
..using `..` for a comment
..DOCUMENT IS THE MINIMAL AMOUNT OF KNOWLEDGE TO USE!!
..I'm assuming that `initial guess for particle positions` is not part of
..optimization

As mentioned in the `Introduction`_, PERI works by fitting a generative model
to the data. Now that you've selected or designed a `Generative Model`_, the
next step is optimizing that model.

In section BLAH, we obtained an initial guess for the particle positions and
radii from centroid methods. We used this initial guess to create a state
``st`` with all the components of the model. We now need to optimize the model.
First, the centroid methods do not provide a guess for global parameters
such as the illumination field or the point-spread function. Second, the
centroid methods frequently miss or mis-feature many particles. Third, we would
like to completely optimize the model to get maximally-accurate position and
radii measurements. In this section, we'll discuss how to optimize a
`peri.states.ImageState` and accomplish all three of these goals.

Initial Optimization
--------------------

The goal of the initial optimization is to fit the image accurately enough
to identify any missing particles. The initial guess for component values such
as the illumination and background fields are usually wildly off. Moreover,
centroid methods frequently return wildly inaccurate particle positions. Both
the global parameters such as the illumination and the local parameters such as
particle positions and radii need to be reasonable before we can find the
missing particles.

`peri.opt.optimize.burn` effectively runs this initial optimization. To do this
initial optimization, simply run
.. code-block:: python

    import peri.opt.optimize as opt
    opt.burn(st, mode='burn', n_loop=3, desc='')

This will usually optimize a state sufficiently enough to add missing particles
to the image. If it does not, you can set ``n_loop`` to a larger value such as
6 or 10. Additionally, ``opt.burn`` saves a copy of the state every so often,
by calling ``peri.states.save(st, desc='burning')``. If you do not want to
save the state every so oftern, set ``desc=None`` in ``opt.burn``; if you want
to save with a different name then set ``desc`` to whatever you want; it will
be passed through to ``peri.states.save``.

Briefly, what does ``opt.burn`` do? The state are optimized by
essentially ``curve-fitting`` with a Levenberg-Marquardt algorithm. However,
a typical state has too many parameters to optimize the entire state at once.
To deal with this, ``opt.burn`` optimizes the parameters in groups, first
optimizing the global parameters such as the illumination and background, then
optimizing the particle parameters. Since these two groups of parameters are
coupled, ``opt.burn`` then alternates back and forth between globals and
particles ``n_loop`` times or until a sufficient convergence is met. Finally,
we've found empirically that it's best to avoid optimizing some parameters of
the model (such as the point-spread function and the particle zscale) until the
end. Calling ``opt.burn`` with ``mode='burn'`` combines all these nuances into
one convenience function.

If you want more details on how ``opt.burn`` functions, see the documentation
or code. If your specific image or model is not optimized well by ``opt.burn``,
or you want additional functionality, then you should look at these functions
which ``opt.burn`` calls or uses:
    `peri.opt.optimize.do_levmarq`_
        Levenberg-Marquardt (LM) optimization on whatever parameter groups
        passed, additionall optimized for large parameter spaces.
    `peri.opt.optimize.do_levmarq_all_particle_groups`_
        LM optimization on all the particles in the image.
    `peri.opt.optimize.do_levmarq_particles`_
        LM optimization on a select number of particles.
    `peri.opt.optimize.LMGlobals`_
        The class that `peri.opt.optimize.do_levmarq` calls to do its optimiz-
        ation. Has more options and attributes which are useful for checking
        convergence.
    `peri.opt.optimize.LMParticleGroupCollection`_
        The class that `peri.opt.optimize.do_levmarq_all_particle_groups` calls
        to do its optimization. Has more options and attributes which are
        useful for checking convergence.
    `peri.opt.optimize.LMParticles`_
        The class that both `peri.opt.optimize.do_levmarq_particles` and
        `peri.opt.optimize.LMParticleGroupCollection` calls to do their optim-
        ization. Has more options and attributes which are useful for checking
        convergence.
    `peri.opt.optimize.LMAugmentedState`_
        Like `LMGlobals` but also allows for effective parameters such as an
        overall radii scale or a radii scale that changes with ``z``.
    `peri.opt.optimize.LMEngine`_
        The workhorse optimizer base class, called by
        `peri.opt.optimize.LMGlobals` and `peri.opt.optimize.LMPartilces`

Add-subtract
------------
Various add-subtract modes

Final Optimization
------------------
opt.burn(mode='burn')
opt.burn(mode='polish')


..save the crap about what a good fit is for later...?

Speeding this process up
------------------------

Good initial guesses
^^^^^^^^^^^^^^^^^^^^

The runner functions
^^^^^^^^^^^^^^^^^^^^

Fitting a small image
^^^^^^^^^^^^^^^^^^^^^

How much of this is necessary?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How Good is my fit? When am I done?
-----------------------------------
uh.... would be nice if you talked about s.sigma

What is a good-fit?
==================

Maybe something about what fitting data is etc

The optimization algorithm: Levenberg-Marquardt
===============================================

Quick Overview of Levenberg-Marquardt
-------------------------------------

Geodesic acceleration in large dimensions
-----------------------------------------

Decimation
----------

Optimizing a State
==================
Still hard to do because of mem....


opt.burn
--------
Alternates between globals and local parameters.

Chapter 1 Title
===============

Section 1.1 Title
-----------------

Subsection 1.1.1 Title
~~~~~~~~~~~~~~~~~~~~~~