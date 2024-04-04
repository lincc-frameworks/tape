.. TAPE documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TAPE (Timeseries Analysis & Processing Engine)
==============================================

A Python package for scalable computing of timeseries data.

TAPE offers a complete ecosystem for loading, filtering, and analyzing
timeseries data. TAPE is built to enable users to run provided and user-defined 
analysis functions at scale in a parallelized and/or distributed manner.

Over the survey lifetime of the `LSST <https://www.lsst.org/about>`_, billions of objects will have multiband lightcurves available, and TAPE has
been built as a framework with the goal of making analysis of LSST-scale
data accessible.

TAPE is built on top of `Dask <https://www.dask.org/>`_, and leverages 
its "lazy evaluation" to only load data and run computations when needed.

How to Use This Guide
==============================================

Begin with the `Getting Started <https://tape.readthedocs.io/en/latest/gettingstarted.html>`_ guide to learn the basics of installation and
walkthrough a simple example of using TAPE.

The `Tutorials <https://tape.readthedocs.io/en/latest/tutorials.html>`_ section showcases the fundamental features of TAPE.

API-level information about TAPE is viewable in the 
`API Reference <https://tape.readthedocs.io/en/latest/autoapi/index.html>`_ section.

Learn more about contributing to this repository in our :doc:`Contribution Guide <gettingstarted/contributing>`.


.. toctree::
   :hidden:

   Home page <self>
   Getting Started <gettingstarted>
   Tutorials <tutorials>
   Examples <examples>
   API Reference <autoapi/index>

