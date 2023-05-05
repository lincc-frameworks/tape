from typing import List

import numpy as np

from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from lsstseries.analysis.structure_function.base_calculator import StructureFunctionCalculator

# MacLeod et al. 2012, Erratum 2014ApJ...782..119M
CONVERSION_TO_SIGMA = 0.74


class Macleod2012StructureFunctionCalculator(StructureFunctionCalculator):
    """This class implements the structure function calculation described in
    MacLeod et al. 2012, 2012ApJ...753..106M [https://arxiv.org/abs/1112.0679]
    MacLeod et al. 2012, Erratum 2014ApJ...782..119M

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
            self._all_d_fluxes.append(d_fluxes)

        dts, sfs = self._calculate_binned_statistics(statistic_to_apply=self.calculate_iqr_sf2_statistic)

        return dts, sfs

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
            Result of calculation defined in MacLeod et al. 2012, Erratum 2014ApJ...782..119M:

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
