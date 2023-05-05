import numpy as np

from lsstseries.analysis.structure_function.macleod_2012.calculator import (
    Macleod2012StructureFunctionCalculator,
)
from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer


def test_basic_calculation():
    """Most basic test possible. Inputs are expected to be 2d numpy arrays."""

    test_t = np.atleast_2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_y = np.atleast_2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    arg_container = StructureFunctionArgumentContainer()
    arg_container.combine = False

    sf_calculator = Macleod2012StructureFunctionCalculator(test_t, test_y, None, arg_container)

    res = sf_calculator.calculate()

    assert res


def test_calculate_iqr_method():
    """This test is specifically for the
    `IqrStructureFunctionCalculator.calculate_iqr_sf2_statistic` method.
    We'll set up an instance of `IqrStructureFunctionCalculator`, but only so
    we can get access to the method we want to test.

    Because the function is only called by scipy.binned_statistics, we can be
    more confident that the input type and shape will be consistent. Thus, there
    are not a lot of unit tests around it to confirm the behavior for unexpected
    inputs.
    """

    test_t = np.atleast_2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_y = np.atleast_2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    arg_container = StructureFunctionArgumentContainer()
    arg_container.combine = False

    sf_calculator = Macleod2012StructureFunctionCalculator(test_t, test_y, None, arg_container)

    test_input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 25th percentile of input = 2.25
    # 75th percentile of input = 6.75
    # 0.74 * (75th - 25th) = 3.33
    output = sf_calculator.calculate_iqr_sf2_statistic(test_input)

    assert output == 3.33
