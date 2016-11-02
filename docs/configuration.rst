*****************
Configuration
*****************

Conf hierarchy
==============

There are four ways to provide options to ``peri`` with a specific order in
which they are checked. The order or importance is currently

1. Command line options
2. Environment variables
3. Configuration file
4. In package defaults

Command line options
--------------------

Some (not all) options can be provided on the command line when running the
executable. These options take precendence above all other configurations. For
example::

    peri feature image.png -vvv

sets the verbosity of logs to level 3.

Environment variables
---------------------

To specify configuration options via environment variables, prepend the
option's name with ``peri-``, replace all dashes ``-`` with underscores ``_``,
and convert to all caps. As an example::

    fftw-wisdom -> PERI_FFTW_WISDOM

These changes are made to make them valid variable names in all shells and to
decrease the chances of a naming collision with other programs. 

The options then can be specified in the shell directly::

    bash: export PERI_OPTION_NAME=value
    csh:  setenv PERI_OPTION_NAME value

or they can be provided on the same line as the executable::

    PERI_OPTION_NAME=value peri feature image.png

Configuration file
------------------

Next after command line options, ``peri`` looks at the json-formatted
configuration file.  By default, this file is located as a hidden file
``.peri.json`` in the user's home folder but can also be specified at a
different location using the environment variable ``PERI_CONF_FILE``.

.. _ref-options-reference:

Options reference
=================

.. automodule:: peri.conf

