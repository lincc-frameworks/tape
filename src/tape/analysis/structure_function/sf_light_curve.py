from typing import Optional
import numpy as np

from tape.analysis.light_curve import LightCurve

MIN_OBSERVATIONS_REQUIRED_FOR_SF = 3


class StructureFunctionLightCurve(LightCurve):
    def __init__(self, times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray):
        super().__init__(times, fluxes, errors, MIN_OBSERVATIONS_REQUIRED_FOR_SF)

        # The derived data from the primary data
        self._all_d_times: np.array = []
        self._all_d_fluxes: np.array = []
        self._all_sum_squared_error: np.array = []

        # _public_ derived data
        self.sample_d_times: np.array = []
        self.sample_d_fluxes: np.array = []
        self.sample_sum_squared_error: np.array = []

        self.number_of_difference_values = None

        self._calculate_differences()

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
        self, number_of_samples: Optional[int] = None, random_generator: Optional[np.random.Generator] = None
    ):
        """Take a random sample of time and flux differences and the sum of squared
        errors. The samples are selected without replacement. The resulting
        sub-sample is not guaranteed to have the same order as the input
        differences.

        Parameters
        ----------
        number_of_samples : int, optional
            Defines the number of samples to be randomly selected from the total
            number of difference values. If not specified, take all of the
            avaliable values
        random_generator: np.random.Generator, optional
            A Numpy random.Generator to sample the lightcurve difference. This
            allows for repeatable random samples to be selected. By default None.

        Raises
        ------
        ValueError
            If samples are requested than are present in the light curve raise
            ValueError.

        """

        if number_of_samples is None:
            number_of_samples = self.number_of_difference_values

        if number_of_samples > self.number_of_difference_values:
            raise ValueError(
                f"Requesting {number_of_samples} samples, but only "
                f"{self.number_of_difference_values} are present in the lightcurve"
            )

        # Initialize a random generator if one was not provided
        if random_generator is None:
            random_generator = np.random.default_rng()

        # Stack the time and flux differences and errors.
        data_block = np.vstack((self._all_d_times, self._all_d_fluxes, self._all_sum_squared_error)).astype(
            float
        )

        # Randomly choose `number_of_samples` from the data_block. Return the
        # random samples into distinct arrays.
        self.sample_d_times, self.sample_d_fluxes, self.sample_sum_squared_error = random_generator.choice(
            data_block, number_of_samples, replace=True, axis=1, shuffle=False
        )
