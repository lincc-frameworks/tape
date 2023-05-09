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
        err2_values = []
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
            self._all_d_fluxes.append(np.abs(d_fluxes))
            err2_values.append(err2s)

        # gather the means of the delta_fluxes per bin
        dts, mean_d_flux_per_bin = self._calculate_binned_statistics(statistic_to_apply="mean")

        # gather the means of the sigma^2 (error) per bin
        _, mean_err2_per_bin = self._calculate_binned_statistics(
            sample_values=err2_values, statistic_to_apply="mean"
        )

        # calculate the structure function
        sfs = np.sqrt(PI_OVER_2 * np.square(mean_d_flux_per_bin) - mean_err2_per_bin)

        return dts, sfs

    @staticmethod
    def name_id() -> str:
        return "bauer_2009b"

    @staticmethod
    def expected_argument_container() -> type:
        return StructureFunctionArgumentContainer
