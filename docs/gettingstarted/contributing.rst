Contribution Guide
==================

Dev Guide - Getting Started
---------------------------

Download code and install dependencies in a conda environment. Run unit tests at the end as a verification that the packages are properly installed.

.. code-block:: bash

    conda create -n seriesenv python=3.11
    conda activate seriesenv

    git clone https://github.com/lincc-frameworks/tape
    cd tape/
    pip install .
    pip install .[dev]  # it may be necessary to use `pip install .'[dev]'` (with single quotes) depending on your machine.

    pip install pytest
    pytest
