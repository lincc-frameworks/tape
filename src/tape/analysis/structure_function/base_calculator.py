from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.stats import binned_statistic

from tape.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer
from tape.analysis.structure_function.sf_light_curve import StructureFunctionLightCurve


class StructureFunctionCalculator(ABC):
    """This is the base class from which all other Structure Function calculator
    methods inherit. Extend this class if you want to create a new Structure
    Function calculation method.
    """

    def __init__(
        self,
        lightcurves: List[StructureFunctionLightCurve],
        argument_container: StructureFunctionArgumentContainer,
    ):
        self._lightcurves = lightcurves
        self._argument_container = argument_container

        self._bins = argument_container.bins  # defaults to None
        self._binning_method = argument_container.bin_method
        self._bin_count_target = argument_container.bin_count_target
        self._equally_weight_lightcurves = argument_container.equally_weight_lightcurves
        self._dts = []
        self._all_d_fluxes = []
        self._sum_error_squared = []
        self._difference_values_per_lightcurve: List[int] = []
        return

    @abstractmethod
    def calculate(self):
        """Abstract method that must be implemented by the child class."""
        raise (NotImplementedError, "Must be implemented by the child class")

    def _bootstrap(self, random_generator=None):
        """This method creates the boostraped samples of difference values"""
        self._get_difference_values_per_lightcurve()

        # if the user defined equal weight in the argument container,
        # use that, otherwise, go to specified number of difference values.
        if self._equally_weight_lightcurves is True:
            least_lightcurve_differences = min(self._difference_values_per_lightcurve)
        else:
            least_lightcurve_differences = self._argument_container.number_lightcurve_samples

        for lc in self._lightcurves:
            lc.select_difference_samples(least_lightcurve_differences, random_generator=random_generator)

    def _get_difference_values_per_lightcurve(self):
        """Retrieves the number of difference values per lightcurve and stores
        them in an array.
        """
        self._difference_values_per_lightcurve = [lc.number_of_difference_values for lc in self._lightcurves]

    def _bin_dts(self, dts):
        """Bin an input array of delta times (dts). Supports several binning
        schemes.

        Parameters
        ----------
        dts : `numpy.ndarray` (N,)
            1-d array of delta times to bin

        Returns
        -------
        bins : `numpy.ndarray` (N,)
            The returned bins array.
        """

        num_bins = int(np.ceil(len(dts) / self._bin_count_target))
        dts_unique = np.unique(dts)
        if self._binning_method == "size":
            quantiles = np.linspace(0.0, 1.0, num_bins + 1)
            self._bins = np.quantile(dts_unique, quantiles)

        elif self._binning_method == "length":
            # Compute num_bins equally spaced bins.
            min_val = dts_unique.min()
            max_val = dts_unique.max()
            self._bins = np.linspace(min_val, max_val, num_bins + 1)

            # Extend the start of the first bin by 0.1% of the range to
            # include the first element. Note this is also done to match
            # Panda's cut function.
            self._bins[0] -= 0.001 * (max_val - min_val)

        elif self._binning_method == "loglength":
            log_vals = np.log(dts_unique)

            # Compute num_bins equally spaced bins in log space.
            min_val = log_vals.min()
            max_val = log_vals.max()
            self._bins = np.linspace(min_val, max_val, num_bins + 1)

            # Extend the start of the first bin by 0.1% of the range to
            # include the first element. Note this is also done to match
            # Panda's cut function.
            self._bins[0] -= 0.001 * (max_val - min_val)
            self._bins = np.exp(self._bins)

        else:
            raise ValueError(f"Method '{self._binning_method}' not recognized")

    def _calculate_binned_statistics(self, sample_values=None, statistic_to_apply="mean"):
        """This method will bin delta_t values stored in `self._dts` using the
        bin edges defined by `self._bins`. Then the corresponding `sample_values`
        in each bin will have a statistic measure applied.

        Parameters
        ----------
        sample_values : `np.ndarray`, optional
            The values that will be used to calculate the `statistic_to_apply`.
            If None or not provided, will use `self._all_d_fluxes` by default.
        statistic_to_apply : str or function, optional
            The statistic to apply to the values in each delta_t bin, by default
            "mean".

        Returns
        -------
        (`List[float]`, `List[float]`)
            A tuple of two lists.
            The first list contains the mean of the delta_t values in each bin.
            The second list contains the result of evaluating the
            statistic measure on the delta_flux values in each delta_t bin.

        Notes
        -----
        1) Largely speaking this is a wrapper over Scipy's `binned_statistic`,
        so any of the statistics supported by that function are valid inputs here.

        2) It is expected that the shapes of `self._dts` and `sample_values` are
        the same. Additionally, any entry at the i_th index of `self._dts` must
        correspond to the same pair of observations as the entry at the i_th
        index of `sample_values`.
        """

        if sample_values is None:
            sample_values = [lc.sample_d_fluxes for lc in self._lightcurves]

        if len(sample_values) != len(self._lightcurves):
            raise AttributeError("Number of lightcurves must equal sample_values.")

        # combining treats all lightcurves as one when calculating the structure function
        if self._argument_container.combine and len(self._lightcurves) > 1:
            all_sample_delta_times = np.hstack([lc.sample_d_times for lc in self._lightcurves])
            all_binning_delta_times = np.hstack([lc._all_d_times for lc in self._lightcurves])
            all_sample_values = np.hstack([s for s in sample_values])

            # binning
            if self._bins is None:
                self._bin_dts(all_binning_delta_times)

            # structure function at specific dt
            # the line below will throw error if the bins are not covering the whole range
            try:
                sfs, _, _ = binned_statistic(
                    all_sample_delta_times, all_sample_values, statistic=statistic_to_apply, bins=self._bins
                )
            except AttributeError:
                raise AttributeError("Length of all_delta_times must equal length of all_sample_value.")

            # return the mean delta_time values for each bin
            bin_means, _, _ = binned_statistic(
                all_sample_delta_times, all_sample_delta_times, statistic="mean", bins=self._bins
            )
            return [bin_means], [sfs]

        # Not combining calculates structure function for each light curve independently
        else:
            # may want to raise warning if len(times) <=1 and combine was set true
            sfs_all = []
            t_all = []
            for lc_idx in range(len(self._lightcurves)):
                # ! double check this to see if this should be len(...) >= 1.
                if len(self._lightcurves[lc_idx]._all_d_times) > 1:
                    # bin the delta_time values, and evaluate the `statistic_to_apply`
                    # for the delta_flux values in each bin.

                    # If the users has not defined bins, calculate them using all the time difference values
                    if self._bins is None:
                        self._bin_dts(self._lightcurves[lc_idx]._all_d_times)

                    try:
                        sfs, _, _ = binned_statistic(
                            self._lightcurves[lc_idx].sample_d_times,
                            sample_values[lc_idx],
                            statistic=statistic_to_apply,
                            bins=self._bins,
                        )
                    except AttributeError:
                        raise AttributeError(
                            "Length of self._lightcurves[lc_idx].sample_d_times array \
                                must equal length of corresponding sample_value array."
                        )

                    sfs_all.append(sfs)

                    # return the mean delta_time values for each bin
                    bin_means, _, _ = binned_statistic(
                        self._lightcurves[lc_idx].sample_d_times,
                        self._lightcurves[lc_idx].sample_d_times,
                        statistic="mean",
                        bins=self._bins,
                    )

                    t_all.append(bin_means)
                else:
                    sfs_all.append(np.array([]))
                    t_all.append(np.array([]))
            return t_all, sfs_all

    @staticmethod
    @abstractmethod
    def name_id() -> str:
        """This method will return the unique name of the Structure Function
        calculation method.
        """
        raise (NotImplementedError, "Must be implemented as a static method by the child class")

    @staticmethod
    @abstractmethod
    def expected_argument_container() -> type:
        """This method will return the argument container class type (not an
        instance) that the Structure Function calculation method requires in
        order to perform it's calculations.
        """
        raise (NotImplementedError, "Must be implemented as a static method by the child class")
