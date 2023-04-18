"""Test TimeSeries analysis functions"""

import numpy as np
import pytest

from lsstseries import TimeSeries, analysis

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
    res = test_series.sf2(sthresh=100)

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
    res = test_series.sf2(sthresh=100)

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
    res = test_series.sf2(sthresh=100)

    assert res["dt"][0] == pytest.approx(4.0, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_dt_bins():
    """
    Test that the binning routines return the expected properties
    """
    # Test on some known data.
    dts = np.array([(201.0 - i) for i in range(200)])

    bins = analysis.structurefunction2._bin_dts(dts, method="size")
    np.testing.assert_allclose(bins, [2.0, 101.5, 201.0])

    bins = analysis.structurefunction2._bin_dts(dts, method="length")
    np.testing.assert_allclose(bins, [1.801, 101.5, 201.0])

    bins = analysis.structurefunction2._bin_dts(dts, method="loglength")
    np.testing.assert_allclose(bins, [1.99080091, 20.04993766, 201.0], rtol=1e-5)

    # Test on large randomized data (with a constant seed).
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


def test_dt_bins_raises_exception():
    """
    Test _bin_dts to make sure it raises an exception for an unknown method.
    """
    # Test on some known data.
    dts = np.array([(201.0 - i) for i in range(2)])
    test_method = "not_a_real_method"

    with pytest.raises(ValueError) as excinfo:
        _ = analysis.structurefunction2._bin_dts(dts, method=test_method)
    assert "Method 'not_a_real_method' not recognized"


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
        lc_id=lc_id,
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        bins=None,
        band_to_calc=None,
        combine=False,
        method="size",
        sthresh=100,
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
        lc_id=lc_id,
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        bins=None,
        band_to_calc=None,
        combine=False,
        method="size",
        sthresh=100,
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
        lc_id=lc_id,
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        bins=None,
        band_to_calc=None,
        combine=False,
        method="size",
        sthresh=100,
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

    res = analysis.calc_sf2(
        lc_id=lc_id,
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        bins=None,
        band_to_calc=test_band_to_calc,
        combine=False,
        method="size",
        sthresh=100,
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
        lc_id=lc_id,
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        bins=None,
        band_to_calc=None,
        combine=False,
        method="size",
        sthresh=100,
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
        lc_id=lc_id,
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        bins=None,
        band_to_calc=None,
        combine=False,
        method="size",
        sthresh=100,
    )

    assert res["dt"][0] == pytest.approx(3.705, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.172482, rel=0.001)
