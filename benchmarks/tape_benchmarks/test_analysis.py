"""Test TimeSeries analysis functions"""

import numpy as np
import pytest

from tape import TimeSeries, analysis
from tape.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer

# pylint: disable=protected-access


def test_stetsonj(benchmark):
    """
    Simple test of StetsonJ function for a known return value
    """

    @benchmark
    def benchmark_method():
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


@pytest.mark.parametrize("lc_id", [None, 1])
def test_sf2_timesseries(benchmark, lc_id):
    @benchmark
    def benchmark_sf2_timesseries():
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
        test_series.sf2()


def test_sf2_timeseries_without_timestamps(benchmark):
    """
    Test of structure function squared function for a known return value without
    providing timestamp values
    """

    @benchmark
    def benchmark_sf2_timeseries_without_timestamps():
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
        test_series.sf2()


def test_sf2_timeseries_with_all_none_timestamps(benchmark):
    """
    Test of structure function squared function for a known return value without
    providing timestamp values
    """

    @benchmark
    def benchmark_method():
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
        test_series.sf2()


def test_sf2_base_case(benchmark):
    """
    Base test case accessing calc_sf2 directly. Does not make use of TimeSeries
    or Ensemble.
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
        test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
        test_band = np.array(["r"] * len(test_y))

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
        )


def test_sf2_base_case_time_as_none_array(benchmark):
    """
    Call calc_sf2 directly. Pass time array of all `None`s.
    Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
        test_t = [None, None, None, None, None, None, None, None]
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
        test_band = np.array(["r"] * len(test_y))

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
        )


def test_sf2_base_case_time_as_none_scalar(benchmark):
    """
    Call calc_sf2 directly. Pass a scalar `None` for time.
    Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
        test_t = None
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
        test_band = np.array(["r"] * len(test_y))

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
        )


def test_sf2_base_case_string_for_band_to_calc(benchmark):
    """
    Base test case accessing calc_sf2 directly. Pass a string for band_to_calc.
    Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
        test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
        test_band = np.array(["r"] * len(test_y))
        test_band_to_calc = "r"

        arg_container = StructureFunctionArgumentContainer()
        arg_container.band_to_calc = test_band_to_calc

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
            argument_container=arg_container,
        )


def test_sf2_base_case_error_as_scalar(benchmark):
    """
    Base test case accessing calc_sf2 directly. Provides a scalar value for
    error. Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
        test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = 0.1
        test_band = np.array(["r"] * len(test_y))

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
        )


def test_sf2_base_case_error_as_none(benchmark):
    """
    Base test case accessing calc_sf2 directly. Provides `None` for error.
    Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
        test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = None
        test_band = np.array(["r"] * len(test_y))

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
        )


def test_sf2_no_lightcurve_ids(benchmark):
    """
    Base test case accessing calc_sf2 directly. Pass no lightcurve ids.
    Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
        test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
        test_band = np.array(["r"] * len(test_y))

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
        )


def test_sf2_no_band_information(benchmark):
    """
    Base test case accessing calc_sf2 directly. Pass no band information
    Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
        test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            lc_id=lc_id,
        )


def test_sf2_least_possible_information(benchmark):
    """
    Base test case accessing calc_sf2 directly. Pass time as None and flux, but
    nothing else.
    Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]

        analysis.calc_sf2(
            time=None,
            flux=test_y,
        )


def test_sf2_least_possible_information_constant_flux(benchmark):
    """
    Base test case accessing calc_sf2 directly. Pass time as None and identical
    flux values, but nothing else.
    Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
        test_y = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        analysis.calc_sf2(time=None, flux=test_y)


# @pytest.mark.skip
def test_sf2_flux_and_band_different_lengths(benchmark):
    """
    Base test case accessing calc_sf2 directly. Flux and band are different
    lengths. Expect an exception to be raised.
    Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
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


def test_sf2_flux_and_lc_id_different_lengths(benchmark):
    """
    Base test case accessing calc_sf2 directly. Flux and lc_id are different
    lengths. Expect an exception to be raised.
    Does not make use of TimeSeries or Ensemble.
    """

    @benchmark
    def benchmark_method():
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


def test_create_arg_container_without_arg_container(benchmark):
    """Base case, no argument container provided, expect a default argument_container
    to be created.
    """

    @benchmark
    def benchmark_method():
        test_sf_method = "basic"
        default_arg_container = StructureFunctionArgumentContainer()

        analysis.structurefunction2._create_arg_container_if_needed(test_sf_method, None)


def test_create_arg_container_with_arg_container(benchmark):
    """Base case, with an argument container provided,
    expect that the argument container will be passed through, untouched.
    """

    @benchmark
    def benchmark_method():
        test_sf_method = "basic"
        default_arg_container = StructureFunctionArgumentContainer()

        # set one property to a non-default value
        default_arg_container.ignore_timestamps = True

        analysis.structurefunction2._create_arg_container_if_needed(test_sf_method, default_arg_container)


def test_validate_band_with_band_value(benchmark):
    """Base case where band would be passed in as a non-default value.
    An argument_container is also provided with a different value. We expect
    that the `band` would be the resulting output.
    """

    @benchmark
    def benchmark_method():
        input_band = ["r"]
        input_flux = [1]
        arg_container = StructureFunctionArgumentContainer()
        arg_container.band = ["b"]

        analysis.structurefunction2._validate_band(input_band, input_flux, arg_container)


def test_validate_band_with_arg_container_band_value(benchmark):
    """Base case where band would be passed in as a default (`None`) value.
    An argument_container is also provided with an actual value. We expect
    that the `arg_container.band` would be the resulting output.
    """

    @benchmark
    def benchmark_method():
        input_band = None
        input_flux = [1]
        arg_container = StructureFunctionArgumentContainer()
        arg_container.band = ["b"]

        analysis.structurefunction2._validate_band(input_band, input_flux, arg_container)


def test_validate_band_with_no_input_values(benchmark):
    """Base case where band is not provided in any location. Expected output is
    an array of 0s equal in length to the input_flux array.
    """

    @benchmark
    def benchmark_method():
        input_band = None
        input_flux = [1]
        arg_container = StructureFunctionArgumentContainer()
        expected_output = np.zeros(len(input_flux), dtype=np.int8)

        analysis.structurefunction2._validate_band(input_band, input_flux, arg_container)


def test_validate_band_with_band_value_wrong_length(benchmark):
    """Band will be passed in, but will be a different length than the input flux."""

    @benchmark
    def benchmark_method():
        input_band = ["r"]
        input_flux = [1, 2]
        arg_container = StructureFunctionArgumentContainer()
        arg_container.band = ["b"]

        with pytest.raises(ValueError) as execinfo:
            analysis.structurefunction2._validate_band(input_band, input_flux, arg_container)


def test_validate_lightcurve_with_lc_value(benchmark):
    """Base case where lc_id would be passed in as a non-default value.
    An argument_container is also provided with a different value. We expect
    that the `lc_id` would be the resulting output.
    """

    @benchmark
    def benchmark_method():
        input_lc_id = [100]
        input_flux = [1]
        arg_container = StructureFunctionArgumentContainer()
        arg_container.lc_id = [333]

        analysis.structurefunction2._validate_lightcurve_id(input_lc_id, input_flux, arg_container)


def test_validate_lightcurve_with_arg_container_lc_value(benchmark):
    """Base case where lc_id would be passed in as a default (`None`) value.
    An argument_container is also provided with an actual value. We expect
    that the `arg_container.lc_id` would be the resulting output.
    """

    @benchmark
    def benchmark_method():
        input_lc_id = None
        input_flux = [1]
        arg_container = StructureFunctionArgumentContainer()
        arg_container.lc_id = [333]

        analysis.structurefunction2._validate_lightcurve_id(input_lc_id, input_flux, arg_container)


def test_validate_lightcurve_with_no_input_values(benchmark):
    """Base case where lc_id is not provided in any location. Expected output is
    an array of 0s equal in length to the input_flux array.
    """

    @benchmark
    def benchmark_method():
        input_lc_id = None
        input_flux = [1]
        arg_container = StructureFunctionArgumentContainer()
        expected_output = np.zeros(len(input_flux), dtype=np.int8)

        analysis.structurefunction2._validate_lightcurve_id(input_lc_id, input_flux, arg_container)


def test_validate_band_with_band_value_wrong_length(benchmark):
    """Lightcurve id will be passed in, but will be a different length than the input flux."""

    @benchmark
    def benchmark_method():
        input_lc_id = [100]
        input_flux = [1, 2]
        arg_container = StructureFunctionArgumentContainer()
        arg_container.lc_id = [333]

        with pytest.raises(ValueError) as execinfo:
            analysis.structurefunction2._validate_band(input_lc_id, input_flux, arg_container)


def test_validate_sf_method_base(benchmark):
    """Will pass in "basic" and expect "basic" and output."""

    @benchmark
    def benchmark_method():
        input_sf_method = "basic"
        arg_container = StructureFunctionArgumentContainer()

        analysis.structurefunction2._validate_sf_method(input_sf_method, arg_container)


def test_validate_sf_method_raises_for_unknown_method(benchmark):
    """Make sure that we raise an exception when an unknown sf_method is provided."""

    @benchmark
    def benchmark_method():
        input_sf_method = "basic"
        arg_container = StructureFunctionArgumentContainer()
        arg_container.sf_method = "bogus_method"

        with pytest.raises(ValueError) as execinfo:
            analysis.structurefunction2._validate_sf_method(input_sf_method, arg_container)


def test_sf2_base_case_macleod_2012(benchmark):
    """
    Base test case accessing calc_sf2 directly. Uses `MacLeod 2012` SF calculation method.
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
        test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
        test_band = np.array(["r"] * len(test_y))
        test_sf_method = "macleod_2012"

        analysis.calc_sf2(
            time=test_t, flux=test_y, err=test_yerr, band=test_band, lc_id=lc_id, sf_method=test_sf_method
        )


def test_sf2_multiple_bands(benchmark):
    """
    Starts with the base case test, but duplicates the test_t, test_y and test_err
    for a second color band.
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
        )


def test_sf2_provide_bins_in_argument_container(benchmark):
    """
    Base test case accessing calc_sf2 directly. Does not make use of TimeSeries
    or Ensemble.
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
        test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
        test_band = np.array(["r"] * len(test_y))

        arg_container = StructureFunctionArgumentContainer()
        arg_container.bins = [0.0, 3.1, 9.0]

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
            argument_container=arg_container,
        )


def test_sf2_with_equal_weighting_one_lightcurve(benchmark):
    """
    Base case of using equal weighting passing only 1 light curve
    """

    @benchmark
    def benchmark_method():
        lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
        test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
        test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
        test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
        test_band = np.array(["r"] * len(test_y))
        test_arg_container = StructureFunctionArgumentContainer()
        test_arg_container.equally_weight_lightcurves = True
        test_arg_container.random_seed = 42

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
            argument_container=test_arg_container,
        )


def test_sf2_with_equal_weighting_multiple_lightcurve(benchmark):
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. Uses a
    predefined random seed for reproducibility.
    """

    @benchmark
    def benchmark_method():
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

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
            argument_container=test_arg_container,
        )


def test_sf2_with_unequal_weighting_multiple_lightcurve(benchmark):
    """
    Passing two light curves with unequal weighting. The first light curve is the
    well tested one, the second has more observations. Uses a
    predefined random seed for reproducibility.
    """

    @benchmark
    def benchmark_method():
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

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
            argument_container=test_arg_container,
        )


def test_sf2_with_equal_weighting_multiple_lightcurve_multiple_samplings(benchmark):
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. We will
    resample multiple times. Uses a predefined random seed for reproducibility.
    """

    @benchmark
    def benchmark_method():
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

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
            argument_container=test_arg_container,
        )


def test_sf2_with_equal_weighting_multiple_lightcurve_multiple_samplings_small_bins(benchmark):
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. We will
    resample multiple times. Uses a predefined random seed for reproducibility.
    """

    @benchmark
    def benchmark_method():
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

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
            argument_container=test_arg_container,
        )


def test_sf2_with_equal_weighting_multiple_lightcurve_multiple_samplings_and_combining(benchmark):
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. We will
    resample multiple times. Also requesting that the results are combined.
    Uses a predefined random seed for reproducibility.
    """

    @benchmark
    def benchmark_method():
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
        test_arg_container.random_seed = 42
        test_arg_container.calculation_repetitions = 3
        test_arg_container.bin_count_target = 4
        test_arg_container.combine = True

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
            argument_container=test_arg_container,
        )


def test_sf2_with_equal_weighting_multiple_lightcurve_multiple_samplings_and_combining_non_default_sigma(
    benchmark,
):
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. We will
    resample multiple times. Also requesting that the results are combined.
    We will use a non-default value for the upper and lower error quantiles.
    Uses a predefined random seed for reproducibility.
    """

    @benchmark
    def benchmark_method():
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
        test_arg_container.random_seed = 42
        test_arg_container.calculation_repetitions = 3
        test_arg_container.bin_count_target = 4
        test_arg_container.combine = True
        test_arg_container.lower_error_quantile = 0.4
        test_arg_container.upper_error_quantile = 0.6

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
            argument_container=test_arg_container,
        )


def test_sf2_with_equal_weighting_multiple_lightcurve_multiple_samplings_small_bins_report_error_seperately(
    benchmark,
):
    """
    Passing two light curves with equal weighting. The first light curve is the
    well tested one, the second is longer so a subsample will be taken. We will
    resample multiple times. Uses a predefined random seed for reproducibility.
    """

    @benchmark
    def benchmark_method():
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
        test_arg_container.report_upper_lower_error_separately = True

        analysis.calc_sf2(
            time=test_t,
            flux=test_y,
            err=test_yerr,
            band=test_band,
            lc_id=lc_id,
            argument_container=test_arg_container,
        )