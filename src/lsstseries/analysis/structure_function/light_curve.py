from typing import Optional
import numpy as np

MIN_OBSERVATIONS_REQUIRED_FOR_SF = 2


class StructureFunctionLightCurve:
    def __init__(self, times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray):
        # The primary data
        self._times = times
        self._fluxes = fluxes
        self._errors = errors

        # The derived data from the primary data
        self._all_d_times: np.array = []
        self._all_d_fluxes: np.array = []
        self._all_sum_squared_error: np.array = []

        # _public_ derived data
        self.sample_d_times: np.array = []
        self.sample_d_fluxes: np.array = []
        self.sample_sum_squared_error: np.array = []

        self.number_of_difference_values = None

        self._process_input_data()
        self._calculate_differences()

    def _process_input_data(self):
        """Cleaning and validation occurs here, ideally by calling
        sub-methods for specific checks and cleaning tasks.
        """
        self._check_input_data_size_is_equal()  # make sure arrays the same length
        self._filter_nans()
        self._check_input_data_length_is_sufficient()  # make sure we have enough values still

    def _filter_nans(self):
        """Mask out any NaN values from time, flux and error arrays"""

        t_mask = np.isnan(self._times)
        f_mask = np.isnan(self._fluxes)
        e_mask = np.isnan(self._errors)
        nan_mask = np.logical_or(t_mask, f_mask, e_mask)

        self._times = self._times[~nan_mask]
        self._fluxes = self._fluxes[~nan_mask]
        self._errors = self._errors[~nan_mask]

    def _check_input_data_size_is_equal(self):
        """Make sure that the three input np.arrays have the same size"""
        if self._times.size != self._fluxes.size or self._times.size != self._errors.size:
            raise ValueError("Input np.arrays are expected to have the same size.")

    def _check_input_data_length_is_sufficient(self):
        """Make sure that we have enough data after cleaning and filtering
        to be able to perform Structure Function calculations.
        """
        if self._times.size < MIN_OBSERVATIONS_REQUIRED_FOR_SF:
            raise ValueError("Too few observations provided to calculate Structure Function differences.")

    def _calculate_differences(self):
        """Calculate the difference between all possible pairs of time and flux.
        Also calculate the sum of all possible pairs of error^2. To avoid
        duplicate values, we filter out any differences that correspond to a
        time difference less than 0.
        """
        dt_matrix = self._times.reshape((1, self._times.size)) - self._times.reshape((self._times.size, 1))
        self._all_d_times = dt_matrix[dt_matrix > 0].flatten()
        self.sample_d_times = self._all_d_times
        self.number_of_difference_values = self._all_d_times.size

        df_matrix = self._fluxes.reshape((1, self._fluxes.size)) - self._fluxes.reshape(
            (self._fluxes.size, 1)
        )
        self._all_d_fluxes = df_matrix[dt_matrix > 0].flatten()
        self.sample_d_fluxes = self._all_d_fluxes

        err2_matrix = (
            self._errors.reshape((1, self._errors.size)) ** 2
            + self._errors.reshape((self._errors.size, 1)) ** 2
        )
        self._all_sum_squared_error = err2_matrix[dt_matrix > 0].flatten()
        self.sample_sum_squared_error = self._all_sum_squared_error

    def select_difference_samples(
        self, number_of_samples: int, random_generator: Optional[np.random.Generator] = None
    ):
        """Take a random sample of time and flux differences and the sum of squared
        errors. The samples are selected without replacement. The resulting
        sub-sample is not guaranteed to have the same order as the input
        differences.

        Parameters
        ----------
        number_of_samples : int
            Defines the number of samples to be randomly selected from the total
            number of difference values.
        random_generator: np.random.Generator, optional
            A Numpy random.Generator to sample the lightcurve difference. This
            allows for repeatable random samples to be selected. By default None

        Raises
        ------
        ValueError
            If samples are requested than are present in the light curve raise
            ValueError.

        Notes
        -----

        """

        if number_of_samples > self.number_of_difference_values:
            raise ValueError(
                f"Requesting {number_of_samples} samples, but only {self.number_of_differences} are present in the lightcurve"
            )

        # Initialize a random generator if one was not provided
        if random_generator is None:
            random_generator = np.random.default_rng()

        # Stack the time and flux differences and errors.
        data_block = np.vstack(
            (self._all_d_times, self._all_d_fluxes, self._all_sum_squared_error), dtype=float
        )

        # Randomly choose `number_of_samples` from the data_block. Return the
        # random samples into distinct arrays.
        self.sample_d_times, self.sample_d_fluxes, self.sample_sum_squared_error = random_generator.choice(
            data_block, number_of_samples, replace=False, axis=1, shuffle=False
        )
