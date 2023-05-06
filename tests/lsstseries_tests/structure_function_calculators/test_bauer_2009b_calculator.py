import numpy as np

from lsstseries.analysis.structure_function.bauer_2009b.calculator import (
    Bauer2009BStructureFunctionCalculator,
)
from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer


def test_basic_calculation():
    """Most basic test possible. Inputs are expected to be 2d numpy arrays."""

    test_t = np.atleast_2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_y = np.atleast_2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_e = np.atleast_2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    arg_container = StructureFunctionArgumentContainer()
    arg_container.combine = False

    sf_calculator = Bauer2009BStructureFunctionCalculator(test_t, test_y, test_e, arg_container)

    res = sf_calculator.calculate()

    assert res
