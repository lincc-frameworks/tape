"""Test timeseries analysis functions"""

from lsstseries import timeseries
import pytest


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
    ts = timeseries()
    test_ts = ts.from_dict(data_dict=test_dict)
    print("test StetsonJ value is: " + str(test_ts.stetson_J()["r"]))
    assert test_ts.stetson_J()["r"] == 0.8


def test_sf2_from_timeseries():
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
    ts = timeseries()
    ts.meta['id'] = 1
    test_ts = ts.from_dict(data_dict=test_dict)
    res = test_ts.sf2(sthresh=100)

    assert res['dt'][0] == pytest.approx(3.705, rel=0.001)
    assert res['sf2'][0] == pytest.approx(0.005365, rel=0.001)
