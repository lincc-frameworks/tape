import numpy as np
import pytest

from lsstseries.analysis.light_curve import LightCurve


def test_lightcurve_creation():
    """Test the creation of a lightcurve, check that the number of
    expected delta values equals expectations.
    """
    test_t = np.array([1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2])
    test_y = np.array([0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2])
    test_yerr = np.array([0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02])

    lc = LightCurve(test_t, test_y, test_yerr)

    assert test_t.size == lc._times.size


def test_lightcurve_removed_nans():
    """Check that NaN values are removed and the data has the expected length."""
    test_t = np.array([1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, np.nan])
    test_y = np.array([0.11, 0.23, 0.45, 0.01, 0.67, 0.32, np.nan, 0.2])
    test_yerr = np.array([0.1, 0.023, 0.045, 0.1, 0.067, np.nan, 0.8, 0.02])

    lc = LightCurve(test_t, test_y, test_yerr)

    assert lc._times.size == test_t.size - 3


def test_lightcurve_validate_same_length():
    """Make sure that we raise a ValueError if the input np.arrays don't
    have the same length
    """
    test_t = np.array([1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2])
    test_y = np.array([0.11, 0.23, 0.45, 0.01])
    test_yerr = np.array([0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02])

    with pytest.raises(ValueError) as excinfo:
        _ = LightCurve(test_t, test_y, test_yerr)

    assert "Input np.arrays are expected to have the same size." in str(excinfo.value)


def test_lightcurve_validate_sufficient_observations():
    """Make sure that we raise an exception if there aren't enough
    observations to calculate Structure Function differences.
    """
    test_t = np.array([1.11])
    test_y = np.array([0.11])
    test_yerr = np.array([0.1])

    with pytest.raises(ValueError) as excinfo:
        _ = LightCurve(test_t, test_y, test_yerr)

    assert "Too few observations provided to calculate Structure Function differences." in str(excinfo.value)
