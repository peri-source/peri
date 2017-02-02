.. role:: python(code)
   :language: python

************
Installation
************

Installation
============

The Python package ``peri`` is compatible with most platforms including Linux,
Mac, and Windows. Currently we are only compatible with Python 2.7.x but very
close to Python 3.x compatibility. In order to run ``peri`` you must have a
modern installation of Python. On all platforms, we recommend using the new
`Anaconda <https://www.continuum.io/downloads>`_.  However, any modern
distribution of Python will work. For example, on Linux, you can you the system
Python installed by default (via ``apt-get install python``). On Mac, you can
use the system Python or one installed from macports / brew.

Quick install
-------------

Official releases of ``peri`` are provided on `PyPI.org
<http://pypi.python.org>`_, a.k.a. the `'cheeseshop'
<https://wiki.python.org/moin/CheeseShop>`_. You can access the source
distribution files there at `peri <https://pypi.python.org/pypi/peri/>`_ or
using the common packaging tools provided with Python. The quickest way is to
run::

    pip install peri

This will install ``peri`` onto your :code:`PYTHONPATH`. Make sure that the
appropriate path is also added to your path so the ``peri`` executable is
available. It can be found in :code:`$PYTHONPATH/bin`.

If you don't have Python, you'll first need to install 64-bit Python,
version 2.7. Download `Python <https://www.python.org/downloads/>`_.

In addition, you'll want 64-bit versions of the following packages
 * ``numpy`` (required)
 * ``scipy`` (required)
 * ``matplotlib`` (required)
 * ``pillow`` (required)
 * ``pyfftw`` (makes calculations *much* faster; all-but-required)
 * ``trackpy`` (useful for analyzing the results and getting initial guesses;
   not required)

Running ``pip install peri`` should install ``peri`` and its dependencies
automatically. Sometimes I have trouble with dependencies on Windows machines.
In these cases, I like to download the dependencies in 64-bit from Christopher
Gohlke's helpful website `here <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.


Source code
-----------

.. warning::

    Development builds of PERI may be unstable and not suitable for production
    use, install at your own risk.

Alternatively, you may install the development version from source. Currently,
the repository is hosted on `GitHub <https://github.com/mattbierbaum/peri>`_.
To download and install from source you can clone and install::

    git clone https://github.com/mattbierbaum/peri.git
    cd peri/
    python setup.py install

After that, given you have properly set your :code:`PATH` variables (see
above), you should have :code:`peri` on the command line.

Source releases are also tagged on GitHub and can be accessed from the cloned
repository. To install a particular release, go to the cloned repo and
checkout the desired tags. These tags can be listed from command line::

    git tag -l
    git checkout <tag name>
    python setup.py install

Contributing
============

If you find that featuring performance is poor for a particular image or you
encounter a bug / issue, please reach out to the developers.

Bugs and issues
---------------

Bugs and issues are currently reported through GitHub `issues
<https://github.com/mattbierbaum/peri/issues>`_. In order to help us as much as
possible to resolve the issue, please enable the highest verbosity in the
``peri`` logging system (:ref:`ref-options-reference`) and upload your logs along with a
link to the image and run script / command line to the issue tracker.

Contributing
------------



