"""Test timeseries analysis functions"""

import numpy as np
import pytest

from lsstseries import analysis, timeseries

# pylint: disable=protected-access


def test_stetsonj():
    """
    Simple test of StetsonJ function for a known return value
    """

    flux_list = [0, 1, 2, 3, 4]
    test_dict = {
        "time": range(len(flux_list)),
        "flux": flux_list,
        "flux_err": [1] * len(flux_list),
        "band": ["r"] * len(flux_list),
    }
    timseries = timeseries()
    test_ts = timseries.from_dict(data_dict=test_dict)
    print("test StetsonJ value is: " + str(test_ts.stetson_J()["r"]))
    assert test_ts.stetson_J()["r"] == 0.8


@pytest.mark.parametrize("lc_id", [None, 1])
def test_sf2_timeseries(lc_id):
    """
    Test of structure function squared function for a known return value
    """

    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]

    test_dict = {
        "time": test_t,
        "flux": test_y,
        "flux_err": test_yerr,
        "band": ["r"] * len(test_y),
    }
    timseries = timeseries()
    timseries.meta["id"] = lc_id
    test_series = timseries.from_dict(data_dict=test_dict)
    res = test_series.sf2(sthresh=100)

    assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_dt_bins():
    """
    Test that the binning routines return the expected properties
    """
    np.random.seed(1)
    dts = np.random.random_sample(1000) * 5 + np.logspace(1, 2, 1000)

    # test size method
    bins = analysis.structurefunction2._bin_dts(dts, method="size")
    binsizes = np.histogram(dts, bins=bins)[0]
    assert len(bins) == 11
    assert len(np.unique(binsizes)) == 1  # Check that all bins are the same size

    # test length method
    bins = analysis.structurefunction2._bin_dts(dts, method="length")
    assert len(bins) == 11

    bins = analysis.structurefunction2._bin_dts(dts, method="loglength")
    assert len(bins) == 11
