import numpy as np
import pytest

from lsstseries.analysis.structure_function.base_calculator import (
    StructureFunctionCalculator,
    register_sf_subclasses,
)
from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from lsstseries.analysis.structure_function.basic.calculator import BasicStructureFunctionCalculator


def test_dt_bins():
    """
    Test that the binning routines return the expected properties
    """

    arg_container = StructureFunctionArgumentContainer()
    sf_method = BasicStructureFunctionCalculator(
        np.atleast_2d([]), np.atleast_2d([]), np.atleast_2d([]), argument_container=arg_container
    )

    # Test on some known data.
    dts = np.array([(201.0 - i) for i in range(200)])

    sf_method._bin_dts(dts)
    bins = sf_method._bins
    np.testing.assert_allclose(bins, [2.0, 101.5, 201.0])

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "length"
    sf_method = BasicStructureFunctionCalculator(
        np.atleast_2d([]), np.atleast_2d([]), np.atleast_2d([]), argument_container=arg_container
    )

    sf_method._bin_dts(dts)
    bins = sf_method._bins
    np.testing.assert_allclose(bins, [1.801, 101.5, 201.0])

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "loglength"
    sf_method = BasicStructureFunctionCalculator(
        np.atleast_2d([]), np.atleast_2d([]), np.atleast_2d([]), argument_container=arg_container
    )

    sf_method._bin_dts(dts)
    bins = sf_method._bins
    np.testing.assert_allclose(bins, [1.99080091, 20.04993766, 201.0], rtol=1e-5)


def test_dt_bins_large_random():
    # Test on large randomized data (with a constant seed).
    np.random.seed(1)
    dts = np.random.random_sample(1000) * 5 + np.logspace(1, 2, 1000)

    arg_container = StructureFunctionArgumentContainer()
    sf_method = BasicStructureFunctionCalculator(
        np.atleast_2d([]), np.atleast_2d([]), np.atleast_2d([]), argument_container=arg_container
    )

    # test size method
    sf_method._bin_dts(dts)
    bins = sf_method._bins
    binsizes = np.histogram(dts, bins=bins)[0]
    assert len(bins) == 11
    assert len(np.unique(binsizes)) == 1  # Check that all bins are the same size

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "length"
    sf_method = BasicStructureFunctionCalculator(
        np.atleast_2d([]), np.atleast_2d([]), np.atleast_2d([]), argument_container=arg_container
    )

    # test length method
    sf_method._bin_dts(dts)
    bins = sf_method._bins
    assert len(bins) == 11

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "loglength"
    sf_method = BasicStructureFunctionCalculator(
        np.atleast_2d([]), np.atleast_2d([]), np.atleast_2d([]), argument_container=arg_container
    )

    sf_method._bin_dts(dts)
    bins = sf_method._bins
    assert len(bins) == 11


def test_dt_bins_raises_exception():
    """
    Test _bin_dts to make sure it raises an exception for an unknown method.
    """
    # Test on some known data.
    dts = np.array([(201.0 - i) for i in range(2)])

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "not_a_real_method"
    sf_method = BasicStructureFunctionCalculator(
        np.atleast_2d([]), np.atleast_2d([]), np.atleast_2d([]), argument_container=arg_container
    )

    with pytest.raises(ValueError) as excinfo:
        _ = sf_method._bin_dts(dts)
    assert "Method 'not_a_real_method' not recognized"


def test_calculate_binned_statistics_raises_top_level():
    """
    Test to ensure that _calculate_binned_statistics will raise an exception if
    self._dts and sample_values do not have the same length along axis=0.
    """

    np.random.seed(1)
    dts = np.random.random_sample(1000) * 5 + np.logspace(1, 2, 1000)

    arg_container = StructureFunctionArgumentContainer()
    sf_method = BasicStructureFunctionCalculator(
        np.atleast_2d([]), np.atleast_2d([]), np.atleast_2d([]), argument_container=arg_container
    )

    sf_method._bin_dts(dts)

    test_sample_values = np.array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(AttributeError) as excinfo:
        _ = sf_method._calculate_binned_statistics(test_sample_values)
    assert "Length of self._dts must equal" in str(excinfo.value)


def test_calculate_binned_statistics_raises_combined():
    """
    Test to ensure that _calculate_binned_statistics will raise an exception if
    self._dts and sample_values do not have the same length when np.hstack'ed.
    """

    np.random.seed(1)
    test_delta_times = np.random.random_sample((3, 10))
    test_sample_values = np.random.random_sample((3, 8))

    arg_container = StructureFunctionArgumentContainer()
    arg_container.combine = True
    sf_method = BasicStructureFunctionCalculator(
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.atleast_2d([]),
        np.atleast_2d([]),
        argument_container=arg_container,
    )

    sf_method._bin_dts(test_delta_times)
    sf_method._dts = test_delta_times

    with pytest.raises(AttributeError) as excinfo:
        _ = sf_method._calculate_binned_statistics(test_sample_values)
    assert "Length of combined self._dts" in str(excinfo.value)


def test_calculate_binned_statistics_raises_individual():
    """
    Test to ensure that _calculate_binned_statistics will raise an exception if
    each of the self._dts and sample_values arrays do not have the same length.
    """

    np.random.seed(1)
    test_delta_times = np.random.random_sample((3, 10))
    test_sample_values = np.random.random_sample((3, 8))

    arg_container = StructureFunctionArgumentContainer()
    sf_method = BasicStructureFunctionCalculator(
        np.atleast_2d([]), np.atleast_2d([]), np.atleast_2d([]), argument_container=arg_container
    )

    sf_method._bin_dts(test_delta_times)
    sf_method._dts = test_delta_times

    with pytest.raises(AttributeError) as excinfo:
        _ = sf_method._calculate_binned_statistics(test_sample_values)
    assert "Length of each self._dts" in str(excinfo.value)


def test_register_sf_subclasses():
    """Base test to ensure that we register the most basic subclass of
    StructureFunctionCalculator
    """
    output = register_sf_subclasses()
    assert output["basic"] == BasicStructureFunctionCalculator


def test_register_sf_subclasses_duplicate_name():
    """Create an child class of StructureFunctionCalculator with an intentionally
    duplicate name to check the assertion of `register_sf_subclasses`.
    """

    class DuplicateStructureFunction(StructureFunctionCalculator):
        def calculate(self):
            return 1

        @staticmethod
        def name_id():
            return "basic"

    with pytest.raises(ValueError) as excinfo:
        _ = register_sf_subclasses()

    assert "Attempted to add duplicate Structure" in str(excinfo.value)
