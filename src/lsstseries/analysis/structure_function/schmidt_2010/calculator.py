import numpy as np

from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from lsstseries.analysis.structure_function.base_calculator import StructureFunctionCalculator

SQRT_PI_OVER_2 = np.sqrt(np.pi / 2.0)


class Schmidt2010StructureFunctionCalculator(StructureFunctionCalculator):
    """This class implements the structure function calculation described in
    Eqn. 2 of Schmidt et al. 2010, 2010ApJ...714.1194S [https://arxiv.org/abs/1002.2642]
    Schmidt et al. 2010, Erratum 2010ApJ...721.1941S

    `SF(delta_t) = mean(sqrt(pi/2) * abs(delta_flux_i,j) - sqrt(err_i^2 + err_j^2))`

    Additional references:
    Graham et al. 2014MNRAS.439..703G [https://arxiv.org/abs/1401.1785]
    """

    def calculate(self):
        values_to_be_binned = []
        for lc_idx in range(len(self._time)):
            lc_times = self._time[lc_idx]
            lc_fluxes = self._flux[lc_idx]
            lc_errors = self._err[lc_idx]

            # mask out any nan values
            t_mask = np.isnan(lc_times)
            f_mask = np.isnan(lc_fluxes)
            e_mask = np.isnan(lc_errors)  # always mask out nan errors?
            lc_mask = np.logical_or(t_mask, f_mask, e_mask)

            lc_times = lc_times[~lc_mask]
            lc_fluxes = lc_fluxes[~lc_mask]
            lc_errors = lc_errors[~lc_mask]

            # compute d_times - difference of times
            dt_matrix = lc_times.reshape((1, lc_times.size)) - lc_times.reshape((lc_times.size, 1))
            d_times = dt_matrix[dt_matrix > 0].flatten()

            # d_fluxes - difference of fluxes
            df_matrix = lc_fluxes.reshape((1, lc_fluxes.size)) - lc_fluxes.reshape((lc_fluxes.size, 1))
            d_fluxes = df_matrix[dt_matrix > 0].flatten()

            # err^2 - errors squared
            err2_matrix = (
                lc_errors.reshape((1, lc_errors.size)) ** 2 + lc_errors.reshape((lc_errors.size, 1)) ** 2
            )
            err2s = err2_matrix[dt_matrix > 0].flatten()

            # build stacks of time and flux differences and errors
            self._dts.append(d_times)

            calculated_values = SQRT_PI_OVER_2 * np.abs(d_fluxes) - np.sqrt(err2s)
            values_to_be_binned.append(calculated_values)

        dts, sfs = self._calculate_binned_statistics(sample_values=values_to_be_binned)

        return dts, sfs

    @staticmethod
    def name_id() -> str:
        return "schmidt_2010"

    @staticmethod
    def expected_argument_container() -> type:
        return StructureFunctionArgumentContainer
