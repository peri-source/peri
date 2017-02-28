*************************
Optimization and sampling
*************************

As mentioned in the :doc:`Walkthrough </walkthrough>`, PERI works by fitting a 
generative model to data. In the :doc:`Walkthrough </walkthrough>`, we obtained 
an initial guess for the particle positions and radii from centroid methods. We 
used this initial guess to create a state ``st`` with all the components of the 
model. We now need to optimize the model. 

First, the centroid methods do not provide a guess for global parameters such 
as the illumination field or the point-spread function. Second, the centroid 
methods frequently miss or mis-feature many particles. Third, we would like to 
completely optimize the model to get maximally-accurate position and radii 
measurements. In this section, we'll discuss how to optimize a
:class:`peri.states.ImageState` and accomplish all three of these goals.

Most of these steps are implemented in the :mod:`peri.runner` convenience 
functions.

Initial Optimization
--------------------

The goal of the initial optimization is to fit the image accurately enough
to identify any missing particles. The initial guess for component values such
as the illumination and background fields are usually wildly off. Moreover,
centroid methods frequently return wildly inaccurate particle positions. Both
the global parameters such as the illumination and the local parameters such as
particle positions and radii need to be reasonable before we can find the
missing particles.

:func:`peri.opt.optimize.burn` effectively runs this initial optimization. To do this
initial optimization, simply run:

.. code-block:: python

    import peri.opt.optimize as opt
    opt.burn(st, mode='burn', n_loop=4, desc='', fractol=0.1)

This will usually optimize a state sufficiently enough to add missing particles
to the image. If it does not, you can set ``n_loop`` to a larger value such as
6 or 10. Additionally, ``opt.burn`` saves a copy of the state every so often,
by calling ``peri.states.save(st, desc='burning')``. If you do not want to
save the state every so oftern, set ``desc=None`` in ``opt.burn``; if you want
to save with a different name then set ``desc`` to whatever you want; it will
be passed through to :func:`peri.states.save`.

Briefly, what does :func:`~peri.opt.optimize.burn` do? The state are optimized by
essentially curve-fitting with a `Levenberg-Marquardt algorithm
<https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_.
However, a typical state has too many parameters to optimize the entire state
at once. To deal with this, ``opt.burn`` optimizes the parameters in groups, first
optimizing the global parameters such as the illumination and background, then
optimizing the particle parameters. Since these two groups of parameters are
coupled, ``opt.burn`` then alternates back and forth between globals and
particles ``n_loop`` times or until a sufficient convergence is met -- here
if the error changes by less than 10% of its best value (``fractol=0.1``) [1]_.
Finally, we've found empirically that it's best to avoid optimizing some
parameters of the model (such as the point-spread function and the particle
zscale) until the end. Calling :func:`~peri.opt.optimize.burn` with 
``mode='burn'`` combines all these nuances into one convenience function [2]_. 

.. [1] If ``burn`` doesn't converge sufficiently and instead stops because
       it has iterated for ``n_loop``, then it will log a warning. For this
       initial optimization we'll ignore it, since we'll keep optimizing the
       state later. However, if you get a warning like this in the later steps,
       it might be worthwile to heed the warning and re-optimize the state.

.. [2] In addition, when the point-spread function or illumination are far
       from their correct values, the particle radii tend to drift to bad
       values before slowly coming back to the correct values. If you have a
       pretty good initial guess for the radii, you can speed this up by
       having ``opt.burn`` ignore the particle radii with the flag
       ``include_rad=False``.

If you want more details on how ``opt.burn`` functions, see the documentation
or code. If your specific image or model is not optimized well by ``opt.burn``,
or you want additional functionality, then you should look at these functions
which ``opt.burn`` calls or uses:

    :func:`peri.opt.optimize.do_levmarq`
        Levenberg-Marquardt (LM) optimization on whatever parameter groups
        passed, additionall optimized for large parameter spaces.
    :func:`peri.opt.optimize.do_levmarq_all_particle_groups`
        LM optimization on all the particles in the image.
    :func:`peri.opt.optimize.do_levmarq_particles`
        LM optimization on a select number of particles.
    :class:`peri.opt.optimize.LMGlobals`
        The class that :func:`peri.opt.optimize.do_levmarq` calls to do its
        optimization. Has more options and attributes which are useful for
        checking convergence.
    :class:`peri.opt.optimize.LMParticleGroupCollection`
        The class that :func:`peri.opt.optimize.do_levmarq_all_particle_groups` calls
        to do its optimization. Has more options and attributes which are
        useful for checking convergence.
    :class:`peri.opt.optimize.LMParticles`
        The class that both :func:`peri.opt.optimize.do_levmarq_particles` and
        :class:`peri.opt.optimize.LMParticleGroupCollection` calls to do their
        optimization. Has more options and attributes which are useful for
        checking convergence.
    :class:`peri.opt.optimize.LMAugmentedState`
        Like :class:`~peri.opt.optimize.LMGlobals` but also allows for effective parameters such as an
        overall radii scale or a radii scale that changes with ``z``.
    :class:`peri.opt.optimize.LMEngine`
        The workhorse optimizer base class, called by
        :class:`~peri.opt.optimize.LMGlobals` and :class:`~peri.opt.optimize.LMParticles`

Add-subtract
------------

After the initial optimization we can add any missing particles and remove any
particles that shouldn't be there. To do this, run:

.. code-block:: python

    import peri.opt.addsubtract as addsub
    num_changed, removed_positions, added_positions = addsub.add_subtract(st,
            rad='calc', min_rad='calc', max_rad='calc', invert=True,
            max_npart='calc')

This function adds missing particles to the image by centroid-featuring the
residuals, with ``invert`` the same as for the initial centroid featuring --
set ``invert=True`` if the particles are dark on a bright background; ``False``
otherwise. In the residuals image, missing particles stick out like sore thumbs
and are easy to find. The function adds a particle at this position with radius
``rad``; setting ``rad='calc'`` makes the function choose the radius internally
as the median radius of all the current particles.

More commonly however, two particles are initially featured as one. The initial
optimization will then split the difference by placing this one particle at a
position between the two particles and giving it a large radius. As a result,
the group of particles gets missed by the centroid featuring and particles are
not added. To combat this, the :func:`~peri.opt.addsub.add_subtract` removes particles that
have a suspiciously large or small radii values, as determined by ``min_rad``
and ``max_rad``. (Setting these two to ``'calc'`` uses the cutoffs at the
median radius +/- 15 radii standard deviations.) With the incorrect large
particles removed, the missing particles can be featured. The function
repeatedly removes bad particles and adds missing particles until either no
change is made or it has iterated over the maximum number of loops.


Main Optimization
-----------------

After adding all the particles, it's time to completely optimize the state. In
my experience, sometimes adding particles causes the globals and the old
particle positions to no longer be correct. To deal with this, run something
like

.. code-block:: python

    opt.burn(st, mode='burn', n_loop=6, desc='', fractol=1e-2)

This usually sets the illumination and particle positions to reasonable values.
At this point, it's time to optimize all the state including the point-spread
function, which we have so far ignored. We can include this with rest of the
parameters with :func:`~peri.opt.optimize.burn` again:

.. code-block:: python

    opt.burn(st, mode='polish', n_loop=6, desc='')

What does this do? First, especially if the initial guess for the point-spread
function was correct, running another optimization with ``mode='burn'`` keeps
the point-spread function from drifting to a bad space because of its strong
coupling with the illumination field. Setting ``mode='polish'`` then causes
burn to optimize everything, alternating between an iteration of optimizing all
the global parameters (including the PSF) and an iteration of optimizing all
the particle positions. Similar to ``mode='burn'``, setting ``mode='polish'``
saves the state after each iteration by calling
``peri.states.save(st, desc='polishing')``; you can set ``desc`` to something
else if you'd like.

Achieving the best-possible state
---------------------------------

Sometimes, after all this, particles are still missing or the fit is still not
perfect. There are still a few more tricks in the ``peri`` package to fix these
problems. These tricks are incorporated in the :func:`peri.runner.finish_state`
for convenience. In case you find a better protocol for your images, here are
the tricks below.

Adding tough missing particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes one pass of :func:`~peri.opt.addsubtract.add_subtract` is not enough
to find all the missing particles, or running the secondary optimizations
reveals that more particles are missing. In these cases, running another 
:func:`~peri.opt.addsubtract.add_subtract` usually fixes the problem and gets
all the particles. However, sometimes there are particles that the normal
:func:`~peri.opt.addsubtract.add_subtract` just can't seem to get right, such
as particles on the edge or tough clusters of particles. For these cases, there
is another function in the :mod:`peri.opt.addsubtract` module:

.. code-block:: python

    num_added, added_positions = addsub.add_subtract_locally(st)

Briefly, :func:`~peri.opt.addsubtract.add_subtract_locally` looks for poorly-
fit regions where the residuals deviate from white Gaussian noise, with the
size of the region roughly set by the optional parameter ``filter_size``, and
the threshold for badness set by ``sigma_cutoff``. The function then removes
*all* the particles in that region and re-adds them based on centroid featuring
again. Since :func:`~peri.opt.addsubtract.add_subtract_locally` removes all the
particles in the region, it's best not to use it until the image is fairly well
fit. Otherwise, the function will attempt to remove nearly all the particles in
the image and re-add them, which takes a long time and will probably fail. That
being said, this function is excellent at fixing doubly-featured particle and
at identifying particles at the edge of or slightly outside of the image. You
can improve its chances of identifying particles at the very edge of the image
by passing a ``minmass`` parameter; I find that ``minmass=0`` frequently works
for particles at the edge of a well-fit state.

Additional Optimizations
^^^^^^^^^^^^^^^^^^^^^^^^

Occasionally the number of optimization loops isn't enough to completely
optimize a state. You can try to fix this by running a few more loops of
:func:`~peri.opt.optimize.burn` with ``mode='burn'`` or ``mode='polish'``,
depending on whether the illumination is sufficiently far from the minimum as
to bias the PSF. But especially for large states, sometimes this is not enough.
In this case use

.. code-block:: python

    opt.finish(st)

:func:`~peri.opt.optimize.burn` makes some numerical approximations in the
least-squares fit to accelerate convegence. However, when the fit is near the
global minimum, sometimes ``burn`` can get stuck.
:func:`~peri.opt.optimize.finish` gets around this, taking a slightly slower
but surer descent to the minimum [3]_.

.. [3] For the algorithm-savvy: A Levenberg-Marquardt algorithm works by
       evaluating the derivative of the each residuals pixel with respect to
       each parameter. A typical microscope image with ``peri`` has a few
       million pixels and a few thousand parameters, making these derivatives
       much too big to store in memory. ``burn`` gets around this by treating
       the particles separately from the microscope parameters, and only
       fitting the microscope parameters with a random subset of the image. But
       taking this random subset causes ``burn`` to get slightly stuck with the
       microscope parameters. ``finish`` gets around this by using the entire
       image and alternating between fitting only a small subset of the
       parameters at once. In addition, by default ``finish`` treats the PSF
       separately, since it is more difficult to fit.

Since I like to be sure that I'm at a global minimum, I always run a few extra
loops of ``opt.burn`` with ``mode='polish'`` or of ``opt.finish`` no matter
what.

What if the optimizer gets stuck? If the optimizer is stuck, and you know you
are not at the minimum, then you can individually optimize certain parameters.
For instance, if you know the PSF is not correct based on the way the residuals
looks, you can specifically optimize the PSF by doing this:

.. code-block:: python

    opt.do_levmarq(st, st.get('psf').params)

or whatever global component you think is poorly optimized. If the error of the
state decreases significantly, then the state was not at the global minimum and
should be sent through another few loops of ``opt.burn`` or ``opt.finish``.

When is the state optimized?
----------------------------

PERI relies on finding a global minimum of the fit. If the fit is not correct,
then obviously your extracted parameters such as positions and radii will not
be correct. How can you check if the state is optimized? Below are a few things
we check to see if a state is optimized. You can find many of these detailed in
the Supplemental Information for our paper.

Checking optimization with the OrthoManipulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The best tool for checking optimization is the :class:`peri.viz.interaction.OrthoManipulator`:

.. code-block:: python

    from peri.viz.interaction import OrthoManipulator
    OrthoManipulator(st)

This will pull up an interactive view of the state, with the data in one panel
and the state model in another. Pressing ``Q`` will cycle the view in the
second panel through the reconstructed model, the fit residuals, and individual
model components. To see if the fit is good, look at the fit residuals. Are
there missing particles, both in the middle of the frame and near the edges?
Can you see shadows of particles? If so, then the state is not optimized. In
contrast, if the residuals are nearly perfect white Gaussian noise, then you're
done.

The :class:`~peri.viz.interaction.OrthoManipulator` has a lot of additional
functionality, including a view of the Fourier transform of the residuals and
the ability to add, remove, or optimize individual particles interactively. Try
it!

Checking optimization by running more optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another way to check is simply to run more loops of ``opt.burn`` or 
``opt.finish``. If the error or the parameters you care about change
significantly, then you probably needed to run more optimization loops. If not,
then you were near the minimum. While doing this for every image is probably
impractical, you can check a few images or a smaller section of an image to
see if your protocol is good.

Seeing if the fitted values are reasonable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Frequently it's possible to tell if the fit is good simply by looking at the
parameters themselves. Do the particle radii vary systematically with ``x``,
``y``, or ``z``? If so, then the image is probbaly not at a good fit. We've
found that variations in ``x`` or ``y`` tend to be due to imperfections in the
ILM, which varies strongly in these directions for us, and variations in ``z``
tend to be due to imperfections in the PSF, due to the increased aberration
with depth. Note that this might not just be a case of a poor fit -- the
complexity of the model could be insufficient. You might need to use a more
realistic PSF or use a higher order for the ILM.

You can do similar checks by looking at either the fitted parameters of the PSF
and other components, or the actual fields themselves using the
:class:`~peri.viz.interaction.OrthoManipulator` or :class:`~peri.viz.interaction.OrthoViewer`.

As an aside, we don't find it terribly useful to check if the residuals are at
the expected level of the noise. If you somehow knew exactly what the noise
level was, then you could check that ``st.residuals.std()`` is what it should
be. However, the difference between a good fit and a poor fit can be one-tenth
of a percent (i.e. 1e-3) of the residuals. It is highly unlikely that you know
the level of the noise to that precision -- the noise level can vary by more
than that from frame-to-frame in a movie due to photobleaching or laser power
fluctuations.

Comparing across Images
^^^^^^^^^^^^^^^^^^^^^^^

Finally, you can compare parameters across images. If you featured multiple
images the same way, and the global parameters differ considerably (by
considerably more than the Cramer-Rao Bound), then the state is either not
fully optimized or the model is incomplete. The same applies if the particle
radii fluctuate considerably from frame-to-frame. You can check this easily
with :func:`peri.test.track.calculate_state_radii_fluctuations`.

Speeding this process up
------------------------

Doing this process from start to finish can take a considerable amount of time.
In addition to the parallelization methods mentioned in `Parallel</parallel>`,
here are a collection of several tricks to finding a good fit faster.

Using a Good Initial Guess; The Runner Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The best method for speeding up the featuring is to use a good initial guess.
If you know the ILM or the PSF accurately, then you can certainly save time by
avoiding the initial fits in `Initial Optimization`_, and possibly even the
final fits in `Main Optimization`_.

``peri`` has convenience functions to use previously-optimized global paramters
in fitting an image. These (and others) are located in the ``runner`` module.
For instance, if you have a previously featured state saved as ``state_name``,
this will feature a new image ``'1.tif'``:

.. code-block:: python

    from peri import runner
    feature_diam = 5  #or whatever feature_diam is best for centroid methods
    actual_rad = 5.38  #the actual radius in pixels of the particles

    st = runner.get_particles_featuring(feature_diam, state_name=state_name,
            im_name='1.tif', actual_rad=actual_rad, do_polish=True)


``runner.get_particles_featuring`` takes all of the global parameters from the
state ``state_name``, switches the image, and re-features an initial guess with
centroid methods. It then optimizes the particle positions and radii before
returning. Setting ``do_polish`` to True will automatically run an
``opt.burn(st, mode='polish')`` on both the globals and particles before
returning the state (takes more time), but this can be omitted for speed. The
state is automatically saved at several points. Similar functionality is
provided by some other of the ``runner`` functions -- for instance,
``runner.translate_featuring`` if the particle positions haven't moved much
between the loaded state and the new image.

Fitting a small image
^^^^^^^^^^^^^^^^^^^^^

The larger the image is, the longer it takes to fit. Fitting a small image
considerably speeds up the fit. You can change the region of the fit by setting
the :class:`peri.util.Tile` of the image, as described in the
:doc:`Walkthrough </walkthrough>`.

Fitting a small image is useful to get a good estimate of global parameters,
especially the point-spread function. Since the exact point-spread functions
included in peri change only with ``z``, fitting a small portion of the image
in ``x`` and ``y`` but over the full ``z`` extent will still give an accurate
PSF.

We highly encourage you do fit a small image very well to find a good PSF. The
PSF is difficult to optimize (its optimization space is far from a simple
quadratic, and there are slow directions in the fit). Highly-optimizing a small
state to get an accurate PSF will do more than save a *lot* of time later. For
larger states the optimizer can even get stuck and terminate, thinking it is at
a good fit when it reality the PSF is far from the minimum, which can severely
bias your fits. Make a small image and optimize it overnight -- say, 50-100
loops of ``opt.burn`` with ``mode='polish'`` to ensure that you've found the
global minimum. You might even want to alternate a loop of burn with a direct
minimization of the PSF, like so:

.. code-block:: python

    import numpy as np
    state_vals = []  #storing to check at the end
    for i in xrange(50):  #or another big number
        opt.burn(st, mode='polish', n_loop=1)
        opt.do_levmarq(st, st.get('psf').params)
        state_vals.append(np.copy(st.state[st.params]))

When it finishes, check that the parameters have stopped changing by plotting
them. For instance, to check the parameter ``psf-alpha``:

.. code-block:: python

    import matplotlib.pyplot as plt
    index = st.params.index('psf-alpha')
    plt.plot(state_vals[:,index])

You should see it smoothly approach a constant value. If it doesn't look
converged, then keep optimizing. If you change your imaging conditions -- the
index of refraction of the solvent or the microscope and lens -- then you will
need to do this again.

Sacrificing Precision for Speed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``peri`` is designed to extract information at the maximum possible accuracy.
It does this by finding the best fit of an accurate model to the data. If you
don't need the maximal possible accuracy, then running ``peri`` to completion
is overkill. For instance, if you just want to distinguish which size a
particle is in a bidisperse suspension, finding the particle radii accurately
to 1 nm is not necessary.

If this is the case, you can save some time by running less optimization loops
or not worrying about finding every last particle. You might also be able to
save time by using a less accurate model -- for instance, you could use an ILM
of lower order to create less parameters to fit, or a less accurate PSF to
decrease the execution time for one model generation. You can find some of
these inexact PSFs in :mod:`peri.comp.psfs`, along with a description of how well
they work in the paper's Supplemental Information.

Inventing a new algorithm for fitting in high-dimensional spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Please do this.
