Installation
============

TAPE is available to install with pip, using the "lf-tape" package name:

.. code-block:: bash

    pip install lf-tape


This will grab the latest release version of TAPE from pip.

Advanced Installation
---------------------

In some cases, installation via pip may not be sufficient. In particular, if you're looking to grab the latest
development version of TAPE, you should instead build TAPE from source. The following process downloads the 
TAPE source code and installs it and any needed dependencies in a fresh conda environment. 

.. code-block:: bash

    conda create -n tape_env python=3.11
    conda activate tape_env

    git clone https://github.com/lincc-frameworks/tape
    cd tape/
    pip install .
    pip install .[dev]  # it may be necessary to use `pip install .'[dev]'` (with single quotes) depending on your machine.

The ``pip install .[dev]`` command is optional, and installs dependencies needed to run the unit tests and build
the documentation. The latest source version of TAPE may be less stable than a release, and so we recommend 
running the unit test suite to verify that your local install is performing as expected.

.. code-block:: bash

    pip install pytest
    pytest
