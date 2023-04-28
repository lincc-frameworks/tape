"""Test TimeSeries analysis functions"""

import numpy as np
import pytest

from lsstseries import TimeSeries, analysis
from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer

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
    timseries = TimeSeries()
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
    timseries = TimeSeries()
    timseries.meta["id"] = lc_id
    test_series = timseries.from_dict(data_dict=test_dict)

    res = test_series.sf2()

    assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_sf2_timeseries_without_timestamps():
    """
    Test of structure function squared function for a known return value without
    providing timestamp values
    """

    test_t = None
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]

    test_dict = {
        "time": test_t,
        "flux": test_y,
        "flux_err": test_yerr,
        "band": ["r"] * len(test_y),
    }
    timeseries = TimeSeries()
    timeseries.meta["id"] = 1
    test_series = timeseries.from_dict(data_dict=test_dict)
    res = test_series.sf2()

    assert res["dt"][0] == pytest.approx(4.0, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_sf2_timeseries_with_all_none_timestamps():
    """
    Test of structure function squared function for a known return value without
    providing timestamp values
    """

    test_t = [None, None, None, None, None, None, None, None]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]

    test_dict = {
        "time": test_t,
        "flux": test_y,
        "flux_err": test_yerr,
        "band": ["r"] * len(test_y),
    }
    timeseries = TimeSeries()
    timeseries.meta["id"] = 1
    test_series = timeseries.from_dict(data_dict=test_dict)
    res = test_series.sf2()

    assert res["dt"][0] == pytest.approx(4.0, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_sf2_base_case():
    """
    Base test case accessing calc_sf2 directly. Does not make use of TimeSeries
    or Ensemble.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_y))

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
    )

    assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_sf2_base_case_time_as_none_array():
    """
    Call calc_sf2 directly. Pass time array of all `None`s.
    Does not make use of TimeSeries or Ensemble.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [None, None, None, None, None, None, None, None]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_y))

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
    )

    assert res["dt"][0] == pytest.approx(4.0, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_sf2_base_case_time_as_none_scalar():
    """
    Call calc_sf2 directly. Pass a scalar `None` for time.
    Does not make use of TimeSeries or Ensemble.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = None
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_y))

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
    )

    assert res["dt"][0] == pytest.approx(4.0, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_sf2_base_case_string_for_band_to_calc():
    """
    Base test case accessing calc_sf2 directly. Pass a string for band_to_calc.
    Does not make use of TimeSeries or Ensemble.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_y))
    test_band_to_calc = "r"

    arg_container = StructureFunctionArgumentContainer()
    arg_container.band_to_calc = test_band_to_calc

    res = analysis.calc_sf2(
        time=test_t, flux=test_y, err=test_yerr, band=test_band, lc_id=lc_id, argument_container=arg_container
    )

    assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_sf2_base_case_error_as_scalar():
    """
    Base test case accessing calc_sf2 directly. Provides a scalar value for
    error. Does not make use of TimeSeries or Ensemble.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = 0.1
    test_band = np.array(["r"] * len(test_y))

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
    )

    assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.152482, rel=0.001)


def test_sf2_base_case_error_as_none():
    """
    Base test case accessing calc_sf2 directly. Provides `None` for error.
    Does not make use of TimeSeries or Ensemble.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = None
    test_band = np.array(["r"] * len(test_y))

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
    )

    assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.172482, rel=0.001)


def test_sf2_no_lightcurve_ids():
    """
    Base test case accessing calc_sf2 directly. Pass no lightcurve ids.
    Does not make use of TimeSeries or Ensemble.
    """
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_y))

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
    )

    assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_sf2_no_band_information():
    """
    Base test case accessing calc_sf2 directly. Pass no band information
    Does not make use of TimeSeries or Ensemble.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        lc_id=lc_id,
    )

    assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_sf2_least_possible_infomation():
    """
    Base test case accessing calc_sf2 directly. Pass time as None and flux, but
    nothing else.
    Does not make use of TimeSeries or Ensemble.
    """
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]

    res = analysis.calc_sf2(
        time=None,
        flux=test_y,
    )

    # assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    # assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)

    assert res["dt"][0] == pytest.approx(4.0, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.172482, rel=0.001)


def test_sf2_least_possible_infomation_constant_flux():
    """
    Base test case accessing calc_sf2 directly. Pass time as None and identical
    flux values, but nothing else.
    Does not make use of TimeSeries or Ensemble.
    """
    test_y = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    res = analysis.calc_sf2(time=None, flux=test_y)

    # assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    # assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)

    assert res["dt"][0] == pytest.approx(4.0, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.0, rel=0.001)
