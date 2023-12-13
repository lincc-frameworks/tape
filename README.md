
<img src="https://github.com/lincc-frameworks/tape/blob/main/docs/DARK_Combo_sm.png?raw=true" width="300" height="100">

# TAPE (Timeseries Analysis & Processing Engine)

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![Documentation Status](https://readthedocs.org/projects/tape/badge/?version=latest)](https://tape.readthedocs.io/en/latest/?badge=latest)
[![Unit test and code coverage](https://github.com/lincc-frameworks/tape/actions/workflows/testing-and-coverage.yml/badge.svg)](https://github.com/lincc-frameworks/tape/actions/workflows/testing-and-coverage.yml)
[![codecov](https://codecov.io/gh/lincc-frameworks/tape/branch/main/graph/badge.svg?token=NFLCNEC55C)](https://codecov.io/gh/lincc-frameworks/tape)

Package for working with LSST time series data

Given the duration and cadence of [Vera C. Rubin LSST](https://www.lsst.org/about), the survey will generate a vast amount of time series information capturing the variability of various objects. Scientists will need flexible and highly scalable tools to store and analyze O(Billions) of time series.  The **Time series Analysis and Processing Engine** (TAPE) is a framework for distributed time series analysis which enables the user to scale their algorithm to LSST data sizes. It allows for efficient and scalable evaluation of algorithms on time domain data through built-in fitting and analysis methods as well as support for user-provided algorithms. TAPE supports ingestion of multiple time series formats, enabling easy access to both LSST time series objects and data from other astronomical surveys.

In short term we are working on two main goals of the project:
  - Enable ease of access to TimeSeries objects in LSST
  - Enable efficient and scalable evaluation of algorithm on time-domain data

This is a LINCC Frameworks project - find more information about LINCC Frameworks [here](https://www.lsstcorporation.org/lincc/frameworks).

To learn about the usage of the package, consult the [Documentation](https://tape.readthedocs.io/en/latest/index.html).

## Installation
TAPE is available to install with pip, using the "lf-tape" package name:
``` 
pip install lf-tape
```

## Getting started - for developers

Download code and install dependencies in a conda environment. Run unit tests at the end as a verification that the packages are properly installed.

```
$ conda create -n seriesenv python=3.10
$ conda activate seriesenv

$ git clone https://github.com/lincc-frameworks/tape
$ cd tape/
$ pip install .
$ pip install .[dev]  # it may be necessary to use `pip install .'[dev]'` (with single quotes) depending on your machine.

$ pip install pytest
$ pytest
```

## Acknowledgements

LINCC Frameworks is supported by Schmidt Futures, a philanthropic initiative founded by Eric and Wendy Schmidt, as part of the Virtual Institute of Astrophysics (VIA).
