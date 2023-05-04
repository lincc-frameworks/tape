from typing import List

import numpy as np
from scipy.stats import binned_statistic

from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from lsstseries.analysis.structure_function.base_calculator import StructureFunctionCalculator

# For details MacLeod et al. 2012, 2012ApJ...753..106M [https://arxiv.org/abs/1112.0679]
CONVERSION_TO_SIGMA = 0.74


class Macleod2012StructureFunctionCalculator(StructureFunctionCalculator):
    """This class implements the structure function calculation described in
    MacLeod et al. 2012, 2012ApJ...753..106M [https://arxiv.org/abs/1112.0679]

    SF_obs(deltaT) = 0.74 * IQR

    Where `IQR` is the interquartile range between 25% and 75% of the sorted
    (y(t) - y(t+delta_t)) distribution.

    References:
    Kozlowski 2016, 2016ApJ...826..118K [https://arxiv.org/abs/1604.05858]
    """

    def __init__(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        err: np.ndarray,
        argument_container: StructureFunctionArgumentContainer,
    ):
        super().__init__(time, flux, err, argument_container)

    def calculate(self):
        all_d_fluxes = []
        for lc_idx in range(len(self._time)):
            lc_times = self._time[lc_idx]
            lc_fluxes = self._flux[lc_idx]

            # mask out any nan values
            t_mask = np.isnan(lc_times)
            f_mask = np.isnan(lc_fluxes)
            lc_mask = np.logical_or(t_mask, f_mask)

            lc_times = lc_times[~lc_mask]
            lc_fluxes = lc_fluxes[~lc_mask]

            # compute difference of times
            dt_matrix = lc_times.reshape((1, lc_times.size)) - lc_times.reshape((lc_times.size, 1))
            d_times = dt_matrix[dt_matrix > 0].flatten()

            # compute difference of fluxes, keep only where difference in time > 0
            df_matrix = lc_fluxes.reshape((1, lc_fluxes.size)) - lc_fluxes.reshape((lc_fluxes.size, 1))
            d_fluxes = df_matrix[dt_matrix > 0].flatten()

            # build stack of times and fluxes
            # `self._dts` and `all_d_fluxes` will have shape = (num_lightcurves, N)
            self._dts.append(d_times)
            all_d_fluxes.append(d_fluxes)

        # combining treats all lightcurves as one when calculating the structure function
        if self._argument_container.combine and len(self._time) > 1:
            self._dts = np.hstack(np.array(self._dts, dtype="object"))
            all_d_fluxes = np.hstack(np.array(all_d_fluxes, dtype="object"))

            # binning
            if self._bins is None:
                self._bin_dts(self._dts)

            # structure function at specific dt
            # the line below will throw error if the bins are not covering the whole range
            sfs, bin_edgs, _ = binned_statistic(
                self._dts, all_d_fluxes, statistic=self.calculate_iqr_sf2_statistic, bins=self._bins
            )
            return [(bin_edgs[0:-1] + bin_edgs[1:]) / 2], [sfs]

        # Not combining calculates structure function for each light curve independently
        else:
            # may want to raise warning if len(times) <=1 and combine was set true
            sfs_all = []
            t_all = []
            for lc_idx in range(len(self._dts)):
                if len(self._dts[lc_idx]) > 1:
                    # binning
                    self._bin_dts(self._dts[lc_idx])
                    sfs, bin_edgs, _ = binned_statistic(
                        self._dts[lc_idx],
                        all_d_fluxes[lc_idx],
                        statistic=self.calculate_iqr_sf2_statistic,
                        bins=self._bins,
                    )
                    sfs_all.append(sfs)
                    t_all.append((bin_edgs[0:-1] + bin_edgs[1:]) / 2)
                else:
                    sfs_all.append(np.array([]))
                    t_all.append(np.array([]))
            return t_all, sfs_all

    def calculate_iqr_sf2_statistic(self, input):
        """For a given set of binned metrics (in this case delta fluxes) calculate
        the interquartile range.

        Parameters
        ----------
        input : `np.ndarray` (N,)
            The delta flux values that correspond to a given delta time bin.

        Returns
        -------
        float
            Result of calculating defined in
            MacLeod et al. 2012, 2012ApJ...753..106M [https://arxiv.org/abs/1112.0679]:

            `SF(dt) = 0.74 * IQR`
        """
        # calculate interquartile range between 25% and 75%.
        iqr = np.subtract(*np.percentile(input, [75, 25]))

        return CONVERSION_TO_SIGMA * iqr

    @staticmethod
    def name_id() -> str:
        return "macleod_2012"

    @staticmethod
    def expected_argument_container() -> type:
        return StructureFunctionArgumentContainer
