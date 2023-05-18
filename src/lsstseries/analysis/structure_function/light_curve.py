import numpy as np


class StructureFunctionLightCurve:
    def __init__(self, times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray):
        # The primary data
        self._times = times
        self._fluxes = fluxes
        self._errors = errors

        # The derived data from the primary data
        self._all_d_times = []
        self._all_d_fluxes = []
        self._all_sum_squared_error = []

        # _public_ derived data
        self.sample_d_times = []
        self.sample_d_fluxes = []
        self.sample_sum_squared_error = []

        self.number_of_difference_values = None

        self._validate_input_data()
        self._clean_input_data()
        self._calculate_differences()

    def _validate_input_data(self):
        """Any validation of the input data occurs here. Currently just confirm
        that the input values have the same size.
        """
        if self._times.size != self._fluxes.size or self._times.size != self._errors.size:
            raise ValueError("Input arrays are expected to have the same size")

    def _clean_input_data(self):
        """Any data cleaning needed should occur here, ideally by calling
        sub-methods for specific cleaning tasks.
        """
        self._filter_nans()

    def _filter_nans(self):
        """Mask out any NaN values from time, flux and error arrays"""

        t_mask = np.isnan(self._times)
        f_mask = np.isnan(self._fluxes)
        e_mask = np.isnan(self._errors)
        nan_mask = np.logical_or(t_mask, f_mask, e_mask)

        self._times = self._times[~nan_mask]
        self._fluxes = self._fluxes[~nan_mask]
        self._errors = self._errors[~nan_mask]

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

    def select_difference_samples(self, number_of_samples, random_generator=None):
        """Take a random sample time and flux differences, and the sum of squared
        errors. The samples will be selected without replacement. The resulting
        sub-sample is not guaranteed to have the same order as the input
        differences.

        Parameters
        ----------
        number_of_samples : int
            Defines the number of samples to be randomly selected from the total
            number of difference values.
        random_generator: np.Generator, optional
            A Numpy random.Generator to sample the lightcurve difference. This
            allows for repeatable random samples to be selected. By default None

        Raises
        ------
        ValueError
            If samples are requested than are present in the light curve raise
            ValueError.
        TypeError
            If the provided random generator is not a `numpy.random.Generator`,
            raise TypeError.

        Notes
        -----

        """

        if number_of_samples > self.number_of_differences:
            raise ValueError(
                f"Requesting {number_of_samples} samples, but only {self.number_of_differences} are present in the lightcurve"
            )

        # Initialize a random generator isf one was not provided
        if random_generator is None:
            random_generator = np.random.default_rng()

        # Check to make sure we have the correct type of generator
        if not isinstance(random_generator, np.random.Generator):
            raise TypeError("Expected a random generator with type: numpy.random.Generator")

        #! Probably makes sense to filter out empty and 0 values here???

        # Stack the time and flux differences and errors.
        data_block = np.vstack(
            (self._all_d_times, self._all_d_fluxes, self._all_sum_squared_error), dtype=float
        )

        # Randomly choose `number_of_samples` from the data_block. Return the
        # random samples into distinct arrays.
        self._sample_d_times, self._sample_d_fluxes, self._sample_sum_squared_error = random_generator.choice(
            data_block, number_of_samples, replace=False, axis=1, shuffle=False
        )
