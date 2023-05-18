from typing import List

import numpy as np

from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from lsstseries.analysis.structure_function.base_calculator import StructureFunctionCalculator
from lsstseries.analysis.structure_function.light_curve import StructureFunctionLightCurve


class BasicStructureFunctionCalculator(StructureFunctionCalculator):
    """
    SF calculation method that calculates excess variance directly as a
    variance of observations with observational errors subtracted.
    For reference, please see Equation 12 in https://arxiv.org/abs/1604.05858
    """

    def __init__(
        self,
        lightcurves: List[StructureFunctionLightCurve],
        argument_container: StructureFunctionArgumentContainer,
    ):
        # The only work done in the __init__ method should be input argument
        # validation. Operating on data should only happen in the `calculate`
        # method.

        super().__init__(lightcurves, argument_container)

    def calculate(self):
        values_to_be_binned = [
            np.square(lc.sample_d_fluxes) - lc.sample_sum_squared_error for lc in self._lightcurves
        ]

        dts, sfs = self._calculate_binned_statistics(sample_values=values_to_be_binned)

        return dts, sfs

    @staticmethod
    def name_id() -> str:
        return "basic"

    @staticmethod
    def expected_argument_container() -> type:
        return StructureFunctionArgumentContainer
