import numpy as np

from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from lsstseries.analysis.structure_function.base_calculator import StructureFunctionCalculator

PI_OVER_2 = np.pi / 2.0


class Bauer2009BStructureFunctionCalculator(StructureFunctionCalculator):
    """This class implements the structure function calculation described in
    Eqn. 5 of Bauer et al. 2009, 2009ApJ...696.1241B [https://arxiv.org/abs/0902.4103]

    `SF(tau) = sqrt( (pi/2) * mean(abs(delta_flux))^2 - mean(err^2) )`

    Additional references:
    Graham et al. 2014MNRAS.439..703G [https://arxiv.org/abs/1401.1785]
    """

    def calculate(self):
        self._compute_difference_arrays()

        # gather the means of the abs(delta_fluxes) per bin
        value_to_be_binned = [np.abs(d_flux) for d_flux in self._all_d_fluxes]
        dts, mean_d_flux_per_bin = self._calculate_binned_statistics(sample_values=value_to_be_binned)

        # gather the means of the sigma^2 (error) per bin
        _, mean_err2_per_bin = self._calculate_binned_statistics(sample_values=self._sum_error_squared)

        # calculate the structure function
        sfs = np.sqrt(PI_OVER_2 * np.square(mean_d_flux_per_bin) - mean_err2_per_bin)

        return dts, sfs

    @staticmethod
    def name_id() -> str:
        return "bauer_2009b"

    @staticmethod
    def expected_argument_container() -> type:
        return StructureFunctionArgumentContainer
