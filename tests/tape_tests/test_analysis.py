"""Test TimeSeries analysis functions"""

import numpy as np
import pytest

from tape import ColumnMapper, Ensemble, TimeSeries, analysis
from tape.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer

# pylint: disable=protected-access


@pytest.mark.parametrize("cls", analysis.AnalysisFunction.__subclasses__())
def test_analysis_function(cls, dask_client):
    """
    Test AnalysisFunction child classes
    """
    # We skip child classes with non-trivial constructors
    try:
        obj = cls()
    except TypeError:
        pytest.skip(f"Class {cls} has non-trivial constructor")

    rows = {
        "id": [8001, 8001, 8001, 8001, 8002, 8002, 8002, 8002, 8002],
        "time": [10.1, 10.2, 10.2, 11.1, 11.2, 11.3, 11.4, 15.0, 15.1],
        "band": ["g", "g", "b", "g", "b", "g", "g", "g", "g"],
        "err": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "flux": [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    }
    cmap = ColumnMapper(id_col="id", time_col="time", flux_col="flux", err_col="err", band_col="band")
    ens = Ensemble(client=False).from_source_dict(rows, column_mapper=cmap)

    assert isinstance(obj.cols(ens), list)
    assert len(obj.cols(ens)) > 0
    assert isinstance(obj.on(ens), list)
    assert len(obj.on(ens)) > 0

    # We assume that there are no mandatory keyword arguments
    result = ens.batch(obj)

    assert len(result) == len(set(rows["id"]))


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

    assert res["dt"][0] == pytest.approx(3.1482, rel=0.001)
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

    assert res["dt"][0] == pytest.approx(3.0, rel=0.001)
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

    assert res["dt"][0] == pytest.approx(3.0, rel=0.001)
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

    assert res["dt"][0] == pytest.approx(3.1482, rel=0.001)
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

    assert res["dt"][0] == pytest.approx(3.0, rel=0.001)
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

    assert res["dt"][0] == pytest.approx(3.0, rel=0.001)
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

    assert res["dt"][0] == pytest.approx(3.1482, rel=0.001)
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

    assert res["dt"][0] == pytest.approx(3.1482, rel=0.001)
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

    assert res["dt"][0] == pytest.approx(3.1482, rel=0.001)
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

    assert res["dt"][0] == pytest.approx(3.1482, rel=0.001)
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

    assert res["dt"][0] == pytest.approx(3.1482, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)


def test_sf2_least_possible_information():
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

    assert res["dt"][0] == pytest.approx(3.0, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.172482, rel=0.001)


def test_sf2_least_possible_information_constant_flux():
    """
    Base test case accessing calc_sf2 directly. Pass time as None and identical
    flux values, but nothing else.
    Does not make use of TimeSeries or Ensemble.
    """
    test_y = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    res = analysis.calc_sf2(time=None, flux=test_y)

    assert res["dt"][0] == pytest.approx(3.0, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.0, rel=0.001)


# @pytest.mark.skip
def test_sf2_flux_and_band_different_lengths():
    """
    Base test case accessing calc_sf2 directly. Flux and band are different
    lengths. Expect an exception to be raised.
    Does not make use of TimeSeries or Ensemble.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2, 0.3]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_t))

    with pytest.raises(ValueError) as execinfo:
        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
        )

    assert "same length" in str(execinfo.value)


def test_sf2_flux_and_lc_id_different_lengths():
    """
    Base test case accessing calc_sf2 directly. Flux and lc_id are different
    lengths. Expect an exception to be raised.
    Does not make use of TimeSeries or Ensemble.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1, 2]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_t))

    with pytest.raises(ValueError) as execinfo:
        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
        )

    assert "Value of `lc_id` should be `None` or array with the same length" in str(execinfo.value)


def test_create_arg_container_without_arg_container():
    """Base case, no argument container provided, expect a default argument_container
    to be created.
    """

    test_sf_method = "basic"
    default_arg_container = StructureFunctionArgumentContainer()

    output_arg_container = analysis.structurefunction2._create_arg_container_if_needed(test_sf_method, None)
    assert default_arg_container.sf_method == output_arg_container.sf_method
    assert default_arg_container.ignore_timestamps == output_arg_container.ignore_timestamps


def test_create_arg_container_with_arg_container():
    """Base case, with an argument container provided,
    expect that the argument container will be passed through, untouched.
    """

    test_sf_method = "basic"
    default_arg_container = StructureFunctionArgumentContainer()

    # set one property to a non-default value
    default_arg_container.ignore_timestamps = True

    output_arg_container = analysis.structurefunction2._create_arg_container_if_needed(
        test_sf_method, default_arg_container
    )
    assert default_arg_container.sf_method == output_arg_container.sf_method
    assert default_arg_container.ignore_timestamps == output_arg_container.ignore_timestamps


def test_validate_band_with_band_value():
    """Base case where band would be passed in as a non-default value.
    An argument_container is also provided with a different value. We expect
    that the `band` would be the resulting output.
    """

    input_band = ["r"]
    input_flux = [1]
    arg_container = StructureFunctionArgumentContainer()
    arg_container.band = ["b"]

    output_band = analysis.structurefunction2._validate_band(input_band, input_flux, arg_container)

    assert output_band == input_band


def test_validate_band_with_arg_container_band_value():
    """Base case where band would be passed in as a default (`None`) value.
    An argument_container is also provided with an actual value. We expect
    that the `arg_container.band` would be the resulting output.
    """

    input_band = None
    input_flux = [1]
    arg_container = StructureFunctionArgumentContainer()
    arg_container.band = ["b"]

    output_band = analysis.structurefunction2._validate_band(input_band, input_flux, arg_container)

    assert output_band == arg_container.band


def test_validate_band_with_no_input_values():
    """Base case where band is not provided in any location. Expected output is
    an array of 0s equal in length to the input_flux array.
    """

    input_band = None
    input_flux = [1]
    arg_container = StructureFunctionArgumentContainer()
    expected_output = np.zeros(len(input_flux), dtype=np.int8)

    output_band = analysis.structurefunction2._validate_band(input_band, input_flux, arg_container)

    assert output_band == expected_output


def test_validate_band_with_band_value_wrong_length():
    """Band will be passed in, but will be a different length than the input flux."""

    input_band = ["r"]
    input_flux = [1, 2]
    arg_container = StructureFunctionArgumentContainer()
    arg_container.band = ["b"]

    with pytest.raises(ValueError) as execinfo:
        analysis.structurefunction2._validate_band(input_band, input_flux, arg_container)

    assert "same length" in str(execinfo.value)


def test_validate_lightcurve_with_lc_value():
    """Base case where lc_id would be passed in as a non-default value.
    An argument_container is also provided with a different value. We expect
    that the `lc_id` would be the resulting output.
    """

    input_lc_id = [100]
    input_flux = [1]
    arg_container = StructureFunctionArgumentContainer()
    arg_container.lc_id = [333]

    output_lc_id = analysis.structurefunction2._validate_lightcurve_id(input_lc_id, input_flux, arg_container)

    assert output_lc_id == input_lc_id


def test_validate_lightcurve_with_arg_container_lc_value():
    """Base case where lc_id would be passed in as a default (`None`) value.
    An argument_container is also provided with an actual value. We expect
    that the `arg_container.lc_id` would be the resulting output.
    """

    input_lc_id = None
    input_flux = [1]
    arg_container = StructureFunctionArgumentContainer()
    arg_container.lc_id = [333]

    output_lc_id = analysis.structurefunction2._validate_lightcurve_id(input_lc_id, input_flux, arg_container)

    assert output_lc_id == arg_container.lc_id


def test_validate_lightcurve_with_no_input_values():
    """Base case where lc_id is not provided in any location. Expected output is
    an array of 0s equal in length to the input_flux array.
    """

    input_lc_id = None
    input_flux = [1]
    arg_container = StructureFunctionArgumentContainer()
    expected_output = np.zeros(len(input_flux), dtype=np.int8)

    output_lc_id = analysis.structurefunction2._validate_lightcurve_id(input_lc_id, input_flux, arg_container)

    assert output_lc_id == expected_output


def test_validate_band_with_band_value_wrong_length():
    """Lightcurve id will be passed in, but will be a different length than the input flux."""

    input_lc_id = [100]
    input_flux = [1, 2]
    arg_container = StructureFunctionArgumentContainer()
    arg_container.lc_id = [333]

    with pytest.raises(ValueError) as execinfo:
        analysis.structurefunction2._validate_band(input_lc_id, input_flux, arg_container)

    assert "same length" in str(execinfo.value)


def test_validate_sf_method_base():
    """Will pass in "basic" and expect "basic" and output."""

    input_sf_method = "basic"
    arg_container = StructureFunctionArgumentContainer()

    output_sf_method = analysis.structurefunction2._validate_sf_method(input_sf_method, arg_container)

    assert input_sf_method == output_sf_method


def test_validate_sf_method_raises_for_unknown_method():
    """Make sure that we raise an exception when an unknown sf_method is provided."""

    input_sf_method = "basic"
    arg_container = StructureFunctionArgumentContainer()
    arg_container.sf_method = "bogus_method"

    with pytest.raises(ValueError) as execinfo:
        analysis.structurefunction2._validate_sf_method(input_sf_method, arg_container)

    assert "Unknown" in str(execinfo.value)


def test_sf2_base_case_macleod_2012():
    """
    Base test case accessing calc_sf2 directly. Uses `MacLeod 2012` SF calculation method.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_y))
    test_sf_method = "macleod_2012"

    res = analysis.calc_sf2(
        time=test_t, flux=test_y, err=test_yerr, band=test_band, lc_id=lc_id, sf_method=test_sf_method
    )

    assert res["dt"][0] == pytest.approx(3.1482, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.1687, rel=0.001)


def test_sf2_multiple_bands():
    """
    Starts with the base case test, but duplicates the test_t, test_y and test_err
    for a second color band.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2, 1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2, 0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
    ]
    test_band = np.array(
        [
            "r",
            "r",
            "r",
            "r",
            "r",
            "r",
            "r",
            "r",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
        ]
    )

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
    )

    assert res["dt"][0] == pytest.approx(3.1482, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)
    assert res["dt"][1] == pytest.approx(3.1482, rel=0.001)
    assert res["sf2"][1] == pytest.approx(0.005365, rel=0.001)


def test_sf2_provide_bins_in_argument_container():
    """
    Base test case accessing calc_sf2 directly. Does not make use of TimeSeries
    or Ensemble.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_y))

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bins = [0.0, 3.1, 9.0]

    res = analysis.calc_sf2(
        time=test_t, flux=test_y, err=test_yerr, band=test_band, lc_id=lc_id, argument_container=arg_container
    )

    assert res["dt"][0] == pytest.approx(1.7581, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.0103, rel=0.001)
    assert res["dt"][1] == pytest.approx(5.0017, rel=0.001)
    assert res["sf2"][1] == pytest.approx(-0.001216, rel=0.001)


def test_sf2_with_random_sampling_one_lightcurve():
    """
    Base case of passing only one light curve
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_y))
    test_arg_container = StructureFunctionArgumentContainer()
    test_arg_container.estimate_err = True
    test_arg_container.random_seed = 42

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
        argument_container=test_arg_container,
    )

    assert res["dt"][0] == pytest.approx(3.4475, rel=0.001)
    assert res["sf2"][0] == pytest.approx(-0.0030716, rel=0.001)


def test_sf2_with_equal_weighting_multiple_lightcurve():
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. Uses a
    predefined random seed for reproducibility.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    test_t = [
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        4.01,
        5.67,
    ]
    test_y = [
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.01,
        0.67,
    ]
    test_yerr = [
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.067,
    ]
    test_band = np.array(["r"] * len(test_y))
    test_arg_container = StructureFunctionArgumentContainer()
    test_arg_container.equally_weight_lightcurves = True
    test_arg_container.random_seed = 42

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
        argument_container=test_arg_container,
    )

    assert res["dt"][0] == pytest.approx(3.148214, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)
    assert res["dt"][1] == pytest.approx(2.891860, rel=0.001)
    assert res["sf2"][1] == pytest.approx(0.048904, rel=0.001)


def test_sf2_with_unequal_weighting_multiple_lightcurve():
    """
    Passing two light curves with unequal weighting. The first light curve is the
    well tested one, the second has more observations. Uses a
    predefined random seed for reproducibility.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    test_t = [
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        4.01,
        5.67,
    ]
    test_y = [
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.01,
        0.67,
    ]
    test_yerr = [
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.067,
    ]
    # test_band = np.array(["r"] * len(test_y))
    test_band = np.array(
        ["r", "r", "r", "r", "r", "g", "g", "g", "r", "r", "r", "r", "r", "r", "g", "g", "g", "g"]
    )
    test_arg_container = StructureFunctionArgumentContainer()
    test_arg_container.equally_weight_lightcurves = False
    test_arg_container.random_seed = 42
    test_arg_container.bin_count_target = 4

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
        argument_container=test_arg_container,
    )
    # change values
    assert res["lc_id"][0] == "1"
    assert res["band"][0] == "g"
    assert res["dt"][0] == pytest.approx(1.2533, rel=0.001)
    assert res["sf2"][0] == pytest.approx(-0.164149, rel=0.001)
    assert res["lc_id"][5] == "2"
    assert res["band"][5] == "r"
    assert res["dt"][5] == pytest.approx(0.8875, rel=0.001)
    assert res["sf2"][5] == pytest.approx(0.0871, rel=0.001)


def test_sf2_with_equal_weighting_multiple_lightcurve_multiple_samplings():
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. We will
    resample multiple times. Uses a predefined random seed for reproducibility.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    test_t = [
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        4.01,
        5.67,
    ]
    test_y = [
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.01,
        0.67,
    ]
    test_yerr = [
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.067,
    ]
    test_band = np.array(["r"] * len(test_y))
    test_arg_container = StructureFunctionArgumentContainer()
    test_arg_container.equally_weight_lightcurves = True
    test_arg_container.random_seed = 42
    test_arg_container.calculation_repetitions = 100

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
        argument_container=test_arg_container,
    )

    assert res["dt"][0] == pytest.approx(3.148214, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.005365, rel=0.001)
    assert res["dt"][1] == pytest.approx(2.891860, rel=0.001)
    assert res["sf2"][1] == pytest.approx(0.048904, rel=0.001)


def test_sf2_with_equal_weighting_multiple_lightcurve_multiple_samplings_small_bins():
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. We will
    resample multiple times. Uses a predefined random seed for reproducibility.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    test_t = [
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        4.01,
        5.67,
    ]
    test_y = [
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.01,
        0.67,
    ]
    test_yerr = [
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.067,
    ]
    test_band = np.array(["r"] * len(test_y))
    test_arg_container = StructureFunctionArgumentContainer()
    test_arg_container.equally_weight_lightcurves = True
    test_arg_container.random_seed = 42
    test_arg_container.calculation_repetitions = 100
    test_arg_container.bin_count_target = 4

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
        argument_container=test_arg_container,
    )

    assert res["dt"][0] == pytest.approx(0.662500, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.031108, rel=0.001)
    assert res["dt"][7] == pytest.approx(0.643333, rel=0.001)
    assert res["sf2"][7] == pytest.approx(0.070499, rel=0.001)


def test_sf2_with_equal_weighting_multiple_lightcurve_multiple_samplings_and_combining():
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. We will
    resample multiple times. Also requesting that the results are combined.
    Uses a predefined random seed for reproducibility.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    test_t = [
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        9.01,
        10.67,
    ]
    test_y = [
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.01,
        0.67,
    ]
    test_yerr = [
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.067,
    ]
    test_band = np.array(["r"] * len(test_y))
    test_arg_container = StructureFunctionArgumentContainer()
    test_arg_container.equally_weight_lightcurves = True
    test_arg_container.estimate_err = True
    test_arg_container.random_seed = 42
    test_arg_container.calculation_repetitions = 3
    test_arg_container.bin_count_target = 4
    test_arg_container.combine = True

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
        argument_container=test_arg_container,
    )

    assert res["dt"][0] == pytest.approx(0.55, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.100116, rel=0.001)
    assert res["1_sigma"][0] == pytest.approx(0.061128, rel=0.001)
    assert res["dt"][9] == pytest.approx(3.035000, rel=0.001)
    assert res["sf2"][9] == pytest.approx(0.039646, rel=0.001)
    assert res["1_sigma"][9] == pytest.approx(0.045012, rel=0.001)


def test_sf2_with_equal_weighting_multiple_lightcurve_multiple_samplings_and_combining_non_default_sigma():
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. We will
    resample multiple times. Also requesting that the results are combined.
    We will use a non-default value for the upper and lower error quantiles.
    Uses a predefined random seed for reproducibility.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    test_t = [
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        9.01,
        10.67,
    ]
    test_y = [
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.01,
        0.67,
    ]
    test_yerr = [
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.067,
    ]
    test_band = np.array(["r"] * len(test_y))
    test_arg_container = StructureFunctionArgumentContainer()
    test_arg_container.equally_weight_lightcurves = True
    test_arg_container.estimate_err = True
    test_arg_container.random_seed = 42
    test_arg_container.calculation_repetitions = 3
    test_arg_container.bin_count_target = 4
    test_arg_container.combine = True
    test_arg_container.lower_error_quantile = 0.4
    test_arg_container.upper_error_quantile = 0.6

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
        argument_container=test_arg_container,
    )

    assert res["dt"][0] == pytest.approx(0.55, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.1001165, rel=0.001)
    assert res["1_sigma"][0] == pytest.approx(0.017979, rel=0.001)
    assert res["dt"][9] == pytest.approx(3.035, rel=0.001)
    assert res["sf2"][9] == pytest.approx(0.0396455, rel=0.001)
    assert res["1_sigma"][9] == pytest.approx(0.0132388, rel=0.001)


def test_sf2_with_equal_weighting_multiple_lightcurve_multiple_samplings_small_bins_report_error_seperately():
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. We will
    resample multiple times. Uses a predefined random seed for reproducibility.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    test_t = [
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        1.11,
        2.23,
        3.45,
        4.01,
        5.67,
        6.32,
        7.88,
        8.2,
        4.01,
        5.67,
    ]
    test_y = [
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.11,
        0.23,
        0.45,
        0.01,
        0.67,
        0.32,
        0.88,
        0.2,
        0.01,
        0.67,
    ]
    test_yerr = [
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.023,
        0.045,
        0.1,
        0.067,
        0.032,
        0.8,
        0.02,
        0.1,
        0.067,
    ]
    test_band = np.array(["r"] * len(test_y))
    test_arg_container = StructureFunctionArgumentContainer()
    test_arg_container.equally_weight_lightcurves = True
    test_arg_container.estimate_err = True
    test_arg_container.random_seed = 42
    test_arg_container.calculation_repetitions = 100
    test_arg_container.bin_count_target = 4
    test_arg_container.report_upper_lower_error_separately = True

    res = analysis.calc_sf2(
        time=test_t,
        flux=test_y,
        err=test_yerr,
        band=test_band,
        lc_id=lc_id,
        argument_container=test_arg_container,
    )

    assert res["dt"][0] == pytest.approx(0.64267, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.040187, rel=0.001)
    assert res["lower_error"][0] == pytest.approx(0.07956, rel=0.001)
    assert res["upper_error"][0] == pytest.approx(0.06126, rel=0.001)
    assert res["dt"][7] == pytest.approx(0.6095, rel=0.001)
    assert res["sf2"][7] == pytest.approx(0.08424, rel=0.001)
    assert res["lower_error"][7] == pytest.approx(0.0962, rel=0.001)
    assert res["upper_error"][7] == pytest.approx(0.051476, rel=0.001)
