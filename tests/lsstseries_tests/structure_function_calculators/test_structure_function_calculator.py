import numpy as np
import pytest

from lsstseries.analysis.structure_function_argument_containers import StructureFunctionArgumentContainer
from lsstseries.analysis.structure_function_calculators import StructureFunctionCalculator


def test_dt_bins():
    """
    Test that the binning routines return the expected properties
    """

    arg_container = StructureFunctionArgumentContainer()
    sf_method = StructureFunctionCalculator([], [], [], argument_container=arg_container)

    # Test on some known data.
    dts = np.array([(201.0 - i) for i in range(200)])

    sf_method._bin_dts(dts)
    bins = sf_method._bins
    np.testing.assert_allclose(bins, [2.0, 101.5, 201.0])

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "length"
    sf_method = StructureFunctionCalculator([], [], [], argument_container=arg_container)

    sf_method._bin_dts(dts)
    bins = sf_method._bins
    np.testing.assert_allclose(bins, [1.801, 101.5, 201.0])

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "loglength"
    sf_method = StructureFunctionCalculator([], [], [], argument_container=arg_container)

    sf_method._bin_dts(dts)
    bins = sf_method._bins
    np.testing.assert_allclose(bins, [1.99080091, 20.04993766, 201.0], rtol=1e-5)


def test_dt_bins_large_random():
    # Test on large randomized data (with a constant seed).
    np.random.seed(1)
    dts = np.random.random_sample(1000) * 5 + np.logspace(1, 2, 1000)

    arg_container = StructureFunctionArgumentContainer()
    sf_method = StructureFunctionCalculator([], [], [], argument_container=arg_container)

    # test size method
    sf_method._bin_dts(dts)
    bins = sf_method._bins
    binsizes = np.histogram(dts, bins=bins)[0]
    assert len(bins) == 11
    assert len(np.unique(binsizes)) == 1  # Check that all bins are the same size

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "length"
    sf_method = StructureFunctionCalculator([], [], [], argument_container=arg_container)

    # test length method
    sf_method._bin_dts(dts)
    bins = sf_method._bins
    assert len(bins) == 11

    arg_container = StructureFunctionArgumentContainer()
    arg_container.bin_method = "loglength"
    sf_method = StructureFunctionCalculator([], [], [], argument_container=arg_container)

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
    sf_method = StructureFunctionCalculator([], [], [], argument_container=arg_container)

    with pytest.raises(ValueError) as excinfo:
        _ = sf_method._bin_dts(dts)
    assert "Method 'not_a_real_method' not recognized"
