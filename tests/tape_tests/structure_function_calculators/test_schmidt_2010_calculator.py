import numpy as np
import pytest

from tape.analysis.structurefunction2 import calc_sf2
from tape.analysis.structure_function.schmidt_2010.calculator import (
    Schmidt2010StructureFunctionCalculator,
)
from tape.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from tape.analysis.structure_function.sf_light_curve import StructureFunctionLightCurve


def test_basic_calculation():
    """Most basic test possible. Inputs are expected to be 2d numpy arrays."""

    test_t = np.atleast_2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_y = np.atleast_2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_e = np.atleast_2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_lc = [StructureFunctionLightCurve(test_t, test_y, test_e)]
    arg_container = StructureFunctionArgumentContainer()
    arg_container.combine = False

    sf_calculator = Schmidt2010StructureFunctionCalculator(test_lc, arg_container)

    res = sf_calculator.calculate()

    assert res


def test_sf2_base_case_schmidt_2010():
    """
    Base test case accessing calc_sf2 directly. Uses `Schmidt 2010` SF calculation method.
    """
    lc_id = [1, 1, 1, 1, 1, 1, 1, 1]
    test_t = [1.11, 2.23, 3.45, 4.01, 5.67, 6.32, 7.88, 8.2]
    test_y = [0.11, 0.23, 0.45, 0.01, 0.67, 0.32, 0.88, 0.2]
    test_yerr = [0.1, 0.023, 0.045, 0.1, 0.067, 0.032, 0.8, 0.02]
    test_band = np.array(["r"] * len(test_y))
    test_sf_method = "schmidt_2010"

    res = calc_sf2(
        time=test_t, flux=test_y, err=test_yerr, band=test_band, lc_id=lc_id, sf_method=test_sf_method
    )

    assert res["dt"][0] == pytest.approx(3.1482, rel=0.001)
    assert res["sf2"][0] == pytest.approx(0.02936347, rel=0.001)
