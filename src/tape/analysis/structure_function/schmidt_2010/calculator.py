import numpy as np

from tape.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from tape.analysis.structure_function.base_calculator import StructureFunctionCalculator

SQRT_PI_OVER_2 = np.sqrt(np.pi / 2.0)


class Schmidt2010StructureFunctionCalculator(StructureFunctionCalculator):
    """This class implements the structure function calculation described in
    Eqn. 2 of Schmidt et al. 2010, 2010ApJ...714.1194S [https://arxiv.org/abs/1002.2642]
    Schmidt et al. 2010, Erratum 2010ApJ...721.1941S

    `SF(delta_t) = mean(sqrt(pi/2) * abs(delta_flux_i,j) - sqrt(err_i^2 + err_j^2))`

    Note that the return value is structure function squared.

    Additional references:
    Graham et al. 2014MNRAS.439..703G [https://arxiv.org/abs/1401.1785]
    """

    def calculate(self):
        values_to_be_binned = [
            SQRT_PI_OVER_2 * np.abs(lc.sample_d_fluxes) - np.sqrt(lc.sample_sum_squared_error)
            for lc in self._lightcurves
        ]

        dts, sfs = self._calculate_binned_statistics(sample_values=values_to_be_binned)
        sfs = [i**2 for i in sfs]

        return dts, sfs

    @staticmethod
    def name_id() -> str:
        return "schmidt_2010"

    @staticmethod
    def expected_argument_container() -> type:
        return StructureFunctionArgumentContainer
