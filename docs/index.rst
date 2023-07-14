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

API-level information about TAPE is viewable in the 
API Reference section.

The Tutorials section showcases the fundamental features of TAPE.

.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
   Tutorials <tutorials>
