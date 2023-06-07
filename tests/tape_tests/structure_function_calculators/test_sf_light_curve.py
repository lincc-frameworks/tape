import numpy as np
import pytest

from tape.analysis.structure_function.sf_light_curve import StructureFunctionLightCurve


def test_lightcurve_creation():
    """Test the creation of a lightcurve, check that the number of
    expected delta values equals expectations.
    """
    test_t = np.array([1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2])
    test_y = np.array([0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2])
    test_yerr = np.array([0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02])

    sf_lc = StructureFunctionLightCurve(test_t, test_y, test_yerr)

    expected_length = test_t.size * (test_t.size - 1) / 2
    assert expected_length == sf_lc._all_d_times.size


def test_lightcurve_sample_equals_all_by_default():
    """When we do not request a sub set of delta values, we expect
    that the `._all_*` arrays will be the same as the `.sample_*`
    arrays.
    """
    test_t = np.array([1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2])
    test_y = np.array([0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2])
    test_yerr = np.array([0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02])

    sf_lc = StructureFunctionLightCurve(test_t, test_y, test_yerr)

    assert sf_lc._all_d_times.size == sf_lc.sample_d_times.size
    assert sf_lc._all_d_fluxes.size == sf_lc.sample_d_fluxes.size
    assert sf_lc._all_sum_squared_error.size == sf_lc.sample_sum_squared_error.size


def test_lightcurve_get_sub_sample():
    """Make sure that sample values have the length requested"""
    test_t = np.array([1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2])
    test_y = np.array([0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2])
    test_yerr = np.array([0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02])

    sf_lc = StructureFunctionLightCurve(test_t, test_y, test_yerr)

    num_samples_requested = 5
    sf_lc.select_difference_samples(num_samples_requested)

    assert sf_lc.sample_d_times.size == num_samples_requested


def test_lightcurve_get_sub_sample_raises():
    """Make sure that the method raises an exception when we request more samples
    than are available."""
    test_t = np.array([1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2])
    test_y = np.array([0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2])
    test_yerr = np.array([0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02])

    sf_lc = StructureFunctionLightCurve(test_t, test_y, test_yerr)

    with pytest.raises(ValueError) as excinfo:
        num_samples_requested = 50000
        _ = sf_lc.select_difference_samples(num_samples_requested)

    assert f"Requesting {num_samples_requested} samples, but only" in str(excinfo.value)


def test_lightcurve_get_sub_sample_with_duplicated_generator():
    """Make sure that the input random generator is respected, and will
    produce repeatable results when reset with the same seed.
    """
    test_t = np.array([1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2])
    test_y = np.array([0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2])
    test_yerr = np.array([0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02])

    sf_lc = StructureFunctionLightCurve(test_t, test_y, test_yerr)

    num_samples_requested = 5
    random_seed = 13

    rng = np.random.default_rng(random_seed)
    sf_lc.select_difference_samples(num_samples_requested, random_generator=rng)
    output_1 = sf_lc.sample_d_fluxes

    rng = np.random.default_rng(random_seed)
    sf_lc.select_difference_samples(num_samples_requested, random_generator=rng)
    output_2 = sf_lc.sample_d_fluxes

    assert np.all(output_1 == output_2)


def test_lightcurve_get_sub_sample_with_same_generator():
    """Make sure the input random generator is respected and produces distinct
    result when run multiple times.
    """
    test_t = np.array([1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2])
    test_y = np.array([0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2])
    test_yerr = np.array([0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02])

    sf_lc = StructureFunctionLightCurve(test_t, test_y, test_yerr)

    num_samples_requested = 5
    random_seed = 13
    rng = np.random.default_rng(random_seed)

    sf_lc.select_difference_samples(num_samples_requested, random_generator=rng)
    output_1 = sf_lc.sample_d_fluxes

    sf_lc.select_difference_samples(num_samples_requested, random_generator=rng)
    output_2 = sf_lc.sample_d_fluxes

    assert not np.all(output_1 == output_2)


def test_lightcurve_validate_sufficient_observations():
    """Make sure that we raise an exception if there aren't enough
    observations to calculate Structure Function differences.
    """
    test_t = np.array([1.11])
    test_y = np.array([0.11])
    test_yerr = np.array([0.1])

    with pytest.raises(ValueError) as excinfo:
        _ = StructureFunctionLightCurve(test_t, test_y, test_yerr)

    assert "Too few observations provided to create `LightCurve`." in str(excinfo.value)
