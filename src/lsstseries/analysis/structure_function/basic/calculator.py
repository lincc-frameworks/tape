from typing import List

import numpy as np
from scipy.stats import binned_statistic

from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from lsstseries.analysis.structure_function.base_calculator import StructureFunctionCalculator


class BasicStructureFunctionCalculator(StructureFunctionCalculator):
    """
    SF calculation method that calculates excess variance directly as a
    variance of observations with observational errors subtracted.
    For reference, please see Equation 12 in https://arxiv.org/abs/1604.05858
    """

    def __init__(
        self,
        time: List[float],
        flux: List[float],
        err: List[float],
        argument_container: StructureFunctionArgumentContainer,
    ):
        # The only work done in the __init__ method should be input argument
        # validation. Operating on data should only happen in the `calculate`
        # method.

        super().__init__(time, flux, err, argument_container)

    def calculate(self):
        cor_flux2_all = []
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

            # compute d_times (difference of times) and
            # d_fluxes (difference of magnitudes, i.e., fluxes) for all gaps
            # d_times - difference of times
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

            # corrected each pair of observations
            cor_flux2 = d_fluxes**2 - err2s

            # build stack of times and fluxes
            self._dts.append(d_times)
            cor_flux2_all.append(cor_flux2)

        # combining treats all lightcurves as one when calculating the structure function
        if self._argument_container.combine and len(self._time) > 1:
            self._dts = np.hstack(np.array(self._dts, dtype="object"))
            cor_flux2_all = np.hstack(np.array(cor_flux2_all, dtype="object"))

            # binning
            if self._bins is None:
                self._bin_dts(self._dts)

            # structure function at specific dt
            # the line below will throw error if the bins are not covering the whole range
            sfs, bin_edgs, _ = binned_statistic(self._dts, cor_flux2_all, "mean", self._bins)
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
                        self._dts[lc_idx], cor_flux2_all[lc_idx], "mean", self._bins
                    )
                    sfs_all.append(sfs)
                    t_all.append((bin_edgs[0:-1] + bin_edgs[1:]) / 2)
                else:
                    sfs_all.append(np.array([]))
                    t_all.append(np.array([]))
            return t_all, sfs_all

    @staticmethod
    def name_id() -> str:
        return "basic"

    @staticmethod
    def expected_argument_container() -> type:
        return StructureFunctionArgumentContainer
