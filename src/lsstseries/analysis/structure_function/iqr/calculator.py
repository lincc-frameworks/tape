from typing import List

import numpy as np

from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from lsstseries.analysis.structure_function.base_calculator import StructureFunctionCalculator

# For details see Kozlowski 16 Equation 10: https://arxiv.org/abs/1604.05858
COEF_CONVERSION_TO_SIGMA = 0.741


class IqrStructureFunctionCalculator(StructureFunctionCalculator):
    """This class implements the structure function calculation described in
    Equation 10 of Kozlowski 16: https://arxiv.org/abs/1604.05858

    SF_obs(deltaT) = 0.741 * IQR

    Where `IQR` is the interquartile range between 25% and 75% of the sorted
    (y(t) - y(t+delta_t)) distribution.
    """

    def __init__(
        self,
        time: List[List[float]],
        flux: List[List[float]],
        err: List[List[float]],
        argument_container: StructureFunctionArgumentContainer,
    ):
        super().__init__(time, flux, err, argument_container)

    def calculate(self):
        sfs_all = []
        t_all = None
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

            # compute difference of fluxes, keep only where difference in time > 0
            df_matrix = lc_fluxes.reshape((1, lc_fluxes.size)) - lc_fluxes.reshape((lc_fluxes.size, 1))
            d_fluxes = df_matrix[dt_matrix > 0].flatten()

            # expect all_d_fluxes to have shape = (num_lightcurves, N)
            all_d_fluxes.append(d_fluxes)

        # treat all data as a single light curve
        if self._argument_container.combine:
            # expect all_d_fluxes to have shape = (1, N)
            all_d_fluxes = np.atleast_2d(np.hstack(all_d_fluxes))

        # calculate interquartile range between 25% and 75%.
        iqr = np.subtract(*np.percentile(all_d_fluxes, [75, 25], axis=1))

        sfs_all = COEF_CONVERSION_TO_SIGMA * iqr

        return t_all, sfs_all

    @staticmethod
    def name_id() -> str:
        return "iqr"

    @staticmethod
    def expected_argument_container() -> type:
        return StructureFunctionArgumentContainer
