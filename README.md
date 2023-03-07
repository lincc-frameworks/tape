
<img src="https://www.lsstcorporation.org/lincc/sites/default/files/PastedGraphic-8.png" width="300" height="100">

# lsstseries

Package for working with LSST time series data

Given the duration and cadence of Rubin LSST, the survey will generate a vast amount of time series information capturing the variability of various objects. Scientists will need flexible and highly scalable tools to store and analyze O(Billions) of time series. Ideally we would like to provide a single unified interface, similar to [RAIL’s](https://lsstdescrail.readthedocs.io/en/latest/index.html) approach for photo-zs, that allows scientists to fit and analyze time series using a variety of methods. This would include implementation of different optimizers, ability to ingest different time series formats, and a set of metrics for comparing model performance (e.g. AIC or Bayes factors).

In short term we are working on two main goals of the project:
  - Enable ease of access to TimeSeries objects in LSST
  - Enable efficient and scalable evaluation of algorithm on time-domain data

This is a LINCC project - find more information about LINCC [here](https://www.lsstcorporation.org/lincc/frameworks)

## Getting started - for developers

Download code and install dependencies in a conda environment. Run unit tests at the end as a verification that the packages are properly installed.

```
$ conda create -n seriesenv python=3.10
$ conda activate seriesenv

$ git clone https://github.com/lincc-frameworks/lsstseries
$ cd lsstseries/
$ pip install .

$ pip install pytest
$ pytest
```
