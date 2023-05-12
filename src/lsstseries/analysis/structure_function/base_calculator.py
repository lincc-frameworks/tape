from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.stats import binned_statistic

from lsstseries.analysis.structure_function.base_argument_container import StructureFunctionArgumentContainer


def register_sf_subclasses():
    """This method will identify all of the subclasses of `StructureFunctionCalculator`
    and build a dictionary that maps `name : subclass`.

    Returns
    -------
    dict
        A dictionary of all of subclasses of `StructureFunctionCalculator`. Where
        the str returned from `subclass.name_id()` is the key, and the class is
        the value.

    Raises
    ------
    ValueError
        If a duplicate key is found, a ValueError will be raised. This would
        likely occur if a user copy/pasted an existing subclass but failed to
        update the unique name_id string.
    """
    subclass_dict = {}
    for subcls in StructureFunctionCalculator.__subclasses__():
        if subcls.name_id() in subclass_dict:
            raise ValueError(
                "Attempted to add duplicate Structure Function calculator name to SF_METHODS: "
                + str(subcls.name_id())
            )

        subclass_dict[subcls.name_id()] = subcls

    return subclass_dict


class StructureFunctionCalculator(ABC):
    """This is the base class from which all other Structure Function calculator
    methods inherit. Extend this class if you want to create a new Structure
    Function calculation method.
    """

    def __init__(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        err: np.ndarray,
        argument_container: StructureFunctionArgumentContainer,
    ):
        self._time = time
        self._flux = flux
        self._err = err
        self._argument_container = argument_container

        self._bins = argument_container.bins
        self._binning_method = argument_container.bin_method
        self._bin_count_target = argument_container.bin_count_target
        self._dts = []
        self._all_d_fluxes = []
        self._sum_error_squared = []
        return

    @abstractmethod
    def calculate(self):
        """Abstract method that must be implemented by the child class."""
        raise (NotImplementedError, "Must be implemented by the child class")

    def _compute_difference_arrays(self):
        """This method handles the calculation of the differences between all
        pairs of values for time and flux. (Additionally the sum of error^2).
        The expected inputs are 2D lists for time, flux and error.

        For each array in time, flux and error, we'll mask out nan's, then use
        the resulting values to calculate the difference of all pairs.

        Then we filter out any differences that correspond to a delta_time <= 0.

        We append the resulting numpy arrays into a corresponding python list,
        and make that available for use in additional calculations.
        """
        for lc_idx in range(len(self._time)):
            # Get one numpy array from the list of numpy arrays
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

            # d_times = difference of times
            dt_matrix = lc_times.reshape((1, lc_times.size)) - lc_times.reshape((lc_times.size, 1))
            d_times = dt_matrix[dt_matrix > 0].flatten()
            self._dts.append(d_times)

            # d_fluxes = difference of fluxes
            df_matrix = lc_fluxes.reshape((1, lc_fluxes.size)) - lc_fluxes.reshape((lc_fluxes.size, 1))
            d_fluxes = df_matrix[dt_matrix > 0].flatten()
            self._all_d_fluxes.append(d_fluxes)

            # err^2 = sum of squared errors
            err2_matrix = (
                lc_errors.reshape((1, lc_errors.size)) ** 2 + lc_errors.reshape((lc_errors.size, 1)) ** 2
            )
            err2s = err2_matrix[dt_matrix > 0].flatten()
            self._sum_error_squared.append(err2s)

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
            sample_values = self._all_d_fluxes

        if len(sample_values) != len(self._dts):
            raise AttributeError("Length of self._dts must equal sample_values.")

        # combining treats all lightcurves as one when calculating the structure function
        if self._argument_container.combine and len(self._time) > 1:
            self._dts = np.hstack(np.array(self._dts, dtype="object"))
            sample_values = np.hstack(np.array(sample_values, dtype="object"))

            # binning
            if self._bins is None:
                self._bin_dts(self._dts)

            # structure function at specific dt
            # the line below will throw error if the bins are not covering the whole range
            try:
                sfs, _, _ = binned_statistic(
                    self._dts, sample_values, statistic=statistic_to_apply, bins=self._bins
                )
            except AttributeError:
                raise AttributeError(
                    "Length of combined self._dts must equal length of combined sample_value."
                )

            # return the mean delta_time values for each bin
            bin_means, _, _ = binned_statistic(self._dts, self._dts, statistic="mean", bins=self._bins)
            return [bin_means], [sfs]

        # Not combining calculates structure function for each light curve independently
        else:
            # may want to raise warning if len(times) <=1 and combine was set true
            sfs_all = []
            t_all = []
            for lc_idx in range(len(self._dts)):
                if len(self._dts[lc_idx]) > 1:
                    # bin the delta_time values, and evaluate the `statistic_to_apply`
                    # for the delta_flux values in each bin.

                    if self._bins is None:
                        self._bin_dts(self._dts[lc_idx])

                    try:
                        sfs, _, _ = binned_statistic(
                            self._dts[lc_idx],
                            sample_values[lc_idx],
                            statistic=statistic_to_apply,
                            bins=self._bins,
                        )
                    except AttributeError:
                        raise AttributeError(
                            "Length of each self._dts array must equal length of corresponding sample_value array."
                        )

                    sfs_all.append(sfs)

                    # return the mean delta_time values for each bin
                    bin_means, _, _ = binned_statistic(
                        self._dts[lc_idx],
                        self._dts[lc_idx],
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
