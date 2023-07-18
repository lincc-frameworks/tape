import numpy as np

from tape.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from tape.analysis.structure_function.base_calculator import StructureFunctionCalculator


class Bauer2009AStructureFunctionCalculator(StructureFunctionCalculator):
    """This class implements the structure function calculation described in
    Eqn. 4 of Bauer et al. 2009, 2009ApJ...696.1241B [https://arxiv.org/abs/0902.4103]

    `SF(tau) = sqrt( mean(delta_flux^2) - mean(err^2) )`

    Note that the return value is structure function squared.

    Additional references:
    Graham et al. 2014MNRAS.439..703G [https://arxiv.org/abs/1401.1785]
    """

    def calculate(self):
        # gather the means of the squared delta_fluxes per bin
        values_to_be_binned = [np.square(lc.sample_d_fluxes) for lc in self._lightcurves]
        dts, mean_d_flux_per_bin = self._calculate_binned_statistics(sample_values=values_to_be_binned)

        # gather the means of the sigma^2 (error) per bin
        values_to_be_binned = [lc.sample_sum_squared_error for lc in self._lightcurves]
        _, mean_err2_per_bin = self._calculate_binned_statistics(sample_values=values_to_be_binned)

        # calculate the structure function squared
        sfs = np.asarray(mean_d_flux_per_bin) - np.asarray(mean_err2_per_bin)

        return dts, sfs

    @staticmethod
    def name_id() -> str:
        return "bauer_2009a"

    @staticmethod
    def expected_argument_container() -> type:
        return StructureFunctionArgumentContainer
