*********************
PERI quickstart guide
*********************


Installation
------------

 * Install 64-bit Python, in version 2.7
 * Install ``numpy``, ``scipy``, ``matplotlib``, ``pyfftw``, and ``pillow``, all for 64 bit Python
 * Install PERI: ``pip install peri``


PERI States
-----------
The centerpiece of ``peri`` is an object known as a ``State`` or ``ImageState``,
as described in detail in :doc:`Architecture </architecture>`. An ``ImageState``
is a complete description of a ``peri``'s fit of an image, containing the raw
data itself, the fitted particle and microscope parameters, the quality of the
fit (the fit error and fit residuals), and even the mathematical model used to
describe the image.

The featured images are stored as ``ImageState`` s, which can be saved and
loaded through ``peri.states.save`` and ``peri.states.load`` . Featuring an
image from start to finish involves creating a state and optimizing all its
parameters. To do this, keep reading....

Featuring an Image, from a Python IDE
-------------------------------------

Open your favorite version of a Python IDE or Jupyter notebook. Type

.. code-block:: python

    from peri import runner

to import ``peri``'s ``runner`` module, which contains most of the convenience
functions for analyzing images. The ``runner`` module contains several functions
which will let you completely feature an image from various starting points:

 * from scratch
 * from a guess of positions and radii
 * using the microscope parameters from another state
 * using the microscope parameters and positions from another state

All of these ``runner`` functions allow you to select the images and
previously-featured states interactively through dialog boxes, for convenience.
If this is not convenient you can instead pass the filenames for the states and images
directly to the runner functions, along with a whole lot more options. Read the
documentation if you want to know more!

...from scratch
~~~~~~~~~~~~~~~
To feature an image of dark spherical particles with a radius of roughly 5
pixels on a bright background, type:

.. code-block:: python

    runner.get_initial_featuring(5)

This will pull up a file dialog box asking you to select the image to feature.
Once you've done this, it will run on its own and save the state to the same
directory as the image.


...from a guess of positions and radii
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perhaps you've already spent a lot of time with another method and have a
pretty good guess for all the particle positions and radii. In that case, run

.. code-block:: python

    runner.feature_from_pos_rad(pos, rad)

where ``pos`` and ``rad`` are ``numpy.ndarray`` s of the particle positions and
radii. Once again, this will pull up a file dialog box asking you to select the
image to feature, and run on its own to produce and save a fully-optimized
state.

...using the microscope parameters from another state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You've already featured a few images from your dataset and have a good
``peri`` state for your microscope. You don't want to spend a ton of time
re-featuring the microscope parameters again; you just want the positions in
the next image. If the particles in the new image have a radius of roughly
5 pixels, run

.. code-block:: python

    runner.get_particles_featuring(5)

You'll select the image and previously-featured state through dialog boxes.

...using the microscope parameters and positions from another state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the particles haven't moved by a whole lot from one frame in your dataset
to the next, then you can use

.. code-block:: python

    runner.translate_featuring()

which also allows for you to select the image through dialog boxes.

Getting your Data
-----------------

* Load a state

.. code-block:: python

    from peri import states
    st = states.load('your-state-name.pkl')  #use your appropriate filename

* Check the state.

.. code-block:: python

    from peri.viz.interaction import OrthoManipulator
    OrthoManipulator(st)  #pulls up an interactive window

``peri`` functions by fitting the image to a detailed model. If the model
doesn't fit the image, then the extracted parameters like particle positions
and radii will be incorrect. The ``OrthoManipulator`` lets you check the fit.
Cycle through various views by hitting ``Q``. If you can see shadows of
particles in the residuals view, or structure in the residuals in Fourier
space, then your fit is far from the minimum. You can improve the fit by
running

.. code-block:: python

    runner.finish_state(st)

See :doc:`Optimization </optimization>` for additional details, tips, and tricks.

* Get the positions and radii:

.. code-block:: python

    positions = st.obj_get_positions()
    radii = st.obj_get_radii()


Missing things:
* tile
* slab, component collection
* starting with a small state