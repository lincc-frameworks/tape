.. TAPE documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TAPE (Timeseries Analysis & Processing Engine)
==============================================

A Python package for scalable computing with LSST timeseries data.

TAPE offers a complete ecosystem for loading, filtering, and analyzing
LSST timeseries data. Over the survey lifetime of the LSST, on order ~billions
of objects will have multiband lightcurves available for analysis. TAPE
is built to enable users to run provided and user-defined analysis functions 
at scale in a parallelized and/or distributed manner.

TAPE is built on top of `Dask <https://www.dask.org/>`_, and leverages 
its "lazy evaluation" to only load data and run computations when needed.

Start with the Getting Started section to learn the basics of installation and
walk through a simple example of using TAPE.

The Tutorials section showcases the fundamental features of TAPE.

API-level information about TAPE is viewable in the 
API Reference section.



Dev Guide - Getting Started
---------------------------

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

.. code-block:: bash

   >> conda create env -n <env_name> python=3.10
   >> conda activate <env_name>


Once you have created a new environment, you can install this project for local
development using the following commands:

.. code-block:: bash

   >> pip install -e .'[dev]'
   >> pre-commit install
   >> conda install pandoc


Notes:

1) The single quotes around ``'[dev]'`` may not be required for your operating system.
2) ``pre-commit install`` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on
   `pre-commit <https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html>`_.
3) Installing ``pandoc`` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   `Sphinx and Python Notebooks <https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks>`_.


.. toctree::
   :hidden:

   Home page <self>
   Getting Started <gettingstarted>
   Tutorials <tutorials>
   Examples <examples>
   API Reference <autoapi/index>

